#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <vector>
#include <xxhash.h>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <tuple>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace ImageTrace {
// Diagnostic only: KYTY_TRACE=WxH[:fmt,fmt...] logs cache events of matching images from flip
// KYTY_TRACE_FROM on (default 0).
std::atomic<uint64_t> g_flip {0};
std::atomic<uint64_t> g_cs_hash {0};
std::atomic<uint32_t> g_buffer_dump_budget {0};

const char* DumpDir() {
	static const char* dir = std::getenv("KYTY_DUMP_DIR");
	return dir;
}
std::atomic<uint64_t> g_cs_groups {0};

struct Filter {
	bool                  enabled = false;
	uint32_t              width   = 0;
	uint32_t              height  = 0;
	std::vector<uint32_t> formats;
	uint64_t              from = 0;
};

const Filter& Get() {
	static const Filter filter = [] {
		Filter      f;
		const char* spec = std::getenv("KYTY_TRACE");
		if (spec == nullptr) {
			return f;
		}
		const char* colon = std::strchr(spec, ':');
		f.enabled = std::sscanf(spec, "%ux%u", &f.width, &f.height) == 2;
		for (const char* p = colon; p != nullptr && *p != '\0'; p = std::strchr(p, ',')) {
			f.formats.push_back(static_cast<uint32_t>(std::strtoul(p + 1, nullptr, 10)));
			p++;
		}
		if (const char* from = std::getenv("KYTY_TRACE_FROM"); from != nullptr) {
			f.from = std::strtoull(from, nullptr, 10);
		}
		return f;
	}();
	return filter;
}

bool Enabled() {
	return Get().enabled && g_flip.load(std::memory_order_relaxed) >= Get().from;
}

bool MatchesExtent(uint32_t width, uint32_t height) {
	return Enabled() && width == Get().width && height == Get().height;
}

bool Matches(const Image& image) {
	const auto& f = Get();
	if (!f.enabled || g_flip.load(std::memory_order_relaxed) < f.from ||
	    image.info.extent.width != f.width || image.info.extent.height != f.height) {
		return false;
	}
	const auto fmt = static_cast<uint32_t>(image.info.pixel_format);
	return f.formats.empty() || std::find(f.formats.begin(), f.formats.end(), fmt) != f.formats.end();
}

void Log(const Image& image, const char* event, const char* extra = "") {
	if (!Matches(image)) {
		return;
	}
	std::printf("TRACE f=%" PRIu64 " s%06" PRIu64 " fmt%u addr=%" PRIx64 "+%" PRIx64
	            " %s gpu=%d buf=%d cpu=%d rt=%d %s\n",
	            g_flip.load(std::memory_order_relaxed), image.serial,
	            static_cast<uint32_t>(image.info.pixel_format), image.info.data.address,
	            image.info.data.size, event, image.IsGpuModified(), image.IsBufferModified(),
	            image.IsCpuDirty(), image.usage.render_target, extra);
}
} // namespace ImageTrace

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 16;
// Largest GPU-written tiled image (in texels) that is written back to guest memory on submit.
constexpr uint64_t SmallTiledReadbackTexels = 64 * 64;

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.info.IsDepth()
	                           ? TextureGetSurfaceFormatInfo(desc.info.guest_format).vk_format
	                           : desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodePackedColorClear(format, metadata.clear_word, clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache), m_readback_linear_images(Config::ReadbackLinearImagesEnabled()),
      m_readback_tiled_images(Config::ReadbackTiledImagesEnabled()),
      m_readback_storage_images(Config::ReadbackStorageImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
		// on a 16 GB card the trigger landed at ~3 GB while DBZ keeps ~5.4 GB
		// live, so the collector evicted 80-100 images a second that the game then recreated.
		// Start collecting at 60% of the budget, still below the pressure level.
		// KYTY_GC_OLD_LIMITS=1 restores the old trigger.
		if (std::getenv("KYTY_GC_OLD_LIMITS") == nullptr) {
			const auto roomy = std::min<int64_t>(budget * 6 / 10,
			    static_cast<int64_t>(m_pressure_gc_memory) - GiB / 2);
			m_trigger_gc_memory = static_cast<uint64_t>(
			    std::max<int64_t>(roomy, static_cast<int64_t>(m_trigger_gc_memory)));
		}
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

// DBZ (UE) places an 11_11_10 float HDR scene target and a 10_10_10_2 UNorm target at the same
// address. Sharing one backing let the 10_10_10_2 writes land in the HDR image, whose readers
// then decoded UNorm bits as packed floats: the rainbow. Packed-float formats get their own
// backing; KYTY_SHARE_PACKED_FLOAT=1 restores sharing.
static bool PackedFloatMismatch(vk::Format a, vk::Format b) {
	static const bool share = std::getenv("KYTY_SHARE_PACKED_FLOAT") != nullptr;
	if (share || a == b) {
		return false;
	}
	const auto packed = [](vk::Format f) {
		return f == vk::Format::eB10G11R11UfloatPack32 || f == vk::Format::eE5B9G9R9UfloatPack32;
	};
	return packed(a) || packed(b);
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (PackedFloatMismatch(cached.pixel_format, requested.pixel_format)) {
		return false;
	}
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return BindingType::Texture;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	RegisterImage(id);
	return id;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	const auto& data  = image.info.data;
	if (image.registered || data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	if (data.size > UINT32_MAX) {
		EXIT("TextureCache: image larger than 4 GiB\n");
	}
	const BucketEntry entry {
	    .address = data.address, .size = static_cast<uint32_t>(data.size), .id = id};
	ForEachPage(data.address, data.size, [this, &entry](uint64_t page) {
		m_image_page_table[page].entries.push_back(entry);
	});
	image.registered = true;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	image.HeapChanged();
	UntrackImage(id);
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* bucket = m_image_page_table.Find(page);
		if (bucket == nullptr) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
		auto&      entries = bucket->entries;
		const auto entry   = std::ranges::find(entries, id, &BucketEntry::id);
		if (entry == entries.end()) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
		entries.erase(entry);
	});
	m_lru_cache.Free(image.lru_id);
	m_total_used_memory -= image.AccountedSize();
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.depth_id) {
		std::erase(m_slot_images[image.depth_id].stencil_ids, id);
	} else {
		// Detach first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image.stencil_ids, {})) {
			FreeImage(association);
		}
	}
	m_download_images.erase(id);
	if (image.info.metadata.kind == ImageMetadataKind::Htile) {
		// Only the last image to acquire an HTile range releases its clear state.
		const auto metadata = m_surface_metas.find(image.info.metadata.range.address);
		if (metadata != m_surface_metas.end() && metadata->second.owner == id) {
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_images.erase(id); });
	} else {
		m_slot_images.erase(id);
	}
}

void TextureCache::FreeImage(ImageId id) {
	ImageTrace::Log(m_slot_images[id], "free");
	m_slot_images[id].ClearGpuModified();
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	// Small tiled targets are written back too: games read GPU results such as reductions and
	// simulation state from them on the CPU (PPSA04263 keeps the origin of its ocean wave grid
	// in a 1x1 R32G32 target, and its water physics read it to sample the wave heights).
	const auto& info   = image.info;
	const auto  texels = static_cast<uint64_t>(info.extent.width) * info.extent.height *
	                     info.extent.depth * info.resources.layers;
	if (info.IsTiled() ? m_readback_tiled_images && texels <= SmallTiledReadbackTexels
	                   : m_readback_linear_images) {
		m_download_images.insert(id);
	}
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	// Converts guest D16 between a four-byte host depth aspect and two-byte color or D16 depth.
	const bool source_wide = DepthAspectTransferBytes(source.backing.format) == sizeof(uint32_t);
	if (source_wide == (DepthAspectTransferBytes(destination.backing.format) == sizeof(uint32_t))) {
		return false;
	}
	auto& depth  = source_wide ? source : destination;
	auto& narrow = source_wide ? destination : source;
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    narrow.info.bytes_per_block != sizeof(uint16_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * sizeof(uint32_t);
	const uint64_t narrow_slice =
	    static_cast<uint64_t>(narrow.info.pitch) * narrow.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || narrow_slice > UINT64_MAX / layers);
	const std::array depth_copies {
	    vk::BufferImageCopy {0,
	                         depth.info.pitch,
	                         depth.info.extent.height,
	                         {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                         {},
	                         depth.info.extent}};
	const std::array narrow_copies {
	    vk::BufferImageCopy {0,
	                         narrow.info.pitch,
	                         narrow.info.extent.height,
	                         {narrow.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
	                                                : vk::ImageAspectFlagBits::eColor,
	                          0, 0, layers},
	                         {},
	                         narrow.info.extent}};

	auto depth_buffer  = m_tiler.GetScratchBuffer(depth_slice * layers);
	auto narrow_buffer = m_tiler.GetScratchBuffer(narrow_slice * layers, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(narrow.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * sizeof(uint32_t),
	    .source_slice_stride = narrow_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_wide) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, narrow_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(narrow_copies, narrow_buffer.buffer, narrow_buffer.offset,
		                   narrow_buffer.size);
	} else {
		source.Download(narrow_copies, narrow_buffer.buffer, narrow_buffer.offset,
		                narrow_buffer.size);
		m_tiler.ConvertD16(narrow_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	destination.RefreshComplete();
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (ImageTrace::Matches(destination)) {
		char extra[96];
		std::snprintf(extra, sizeof(extra), "from s%06" PRIu64 " fmt%u", source.serial,
		              static_cast<uint32_t>(source.info.pixel_format));
		ImageTrace::Log(destination, "copy_image", extra);
	}
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	if (m_slot_images[destination_id].IsBufferModified()) {
		RefreshImage(destination_id);
	}
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	// Depth bytes sampled through a color format no depth view provides (16_UINT, 8_8_UNORM)
	// need a color copy of the whole single-sample depth layout.
	const bool raw_depth_texture =
	    cached.info.IsDepth() && !requested.IsDepth() && !cached.info.HasStencil() &&
	    !requested.HasMetadata() && cached.backing.samples == 1 && requested.samples == 1 &&
	    requested.pitch == cached.info.pitch &&
	    (requested.resources == cached.info.resources || retain_cached_layout) &&
	    !IsSupportedSampledDepthFormat(cached.info.pixel_format, requested.pixel_format);
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_depth_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout          = cached.info.mip_layout;
		info.tiling              = cached.info.tiling;
		info.tiled_slice_stride  = cached.info.tiled_slice_stride;
		info.linear_slice_stride = cached.info.linear_slice_stride;
		info.first_tail_level    = cached.info.first_tail_level;
	}
	info.htile_initial_clear  = false;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

namespace BcAlias {

bool Matches(const ImageInfo& requested, const ImageInfo& cached) {
	return requested.guest_format != cached.guest_format &&
	       requested.IsBlock() != cached.IsBlock() &&
	       requested.tile_mode == cached.tile_mode &&
	       requested.bytes_per_block == cached.bytes_per_block &&
	       requested.samples == 1 && cached.samples == 1 &&
	       (requested.data != cached.data || requested.resources != cached.resources ||
	        requested.BlockExtent() != cached.BlockExtent()) &&
	       !TextureCache::SameBacking(cached, requested, false);
}

bool Publish(TextureCache& cache, Image& image) {
	if (!cache.CanDownload(image)) {
		return false;
	}
	// ObtainBuffer preserves padding and records ownership before the GPU-only transfer.
	auto [buffer, offset] =
	    cache.m_buffer_cache.ObtainBuffer(image.info.data.address, image.info.data.size, true);
	cache.DownloadImage(image, *buffer, offset, image.info.data.size, image.info.resources.levels);
	image.ClearGpuModified();
	image.MarkBufferModified();
	return true;
}

void Resolve(TextureCache& cache, ImageId cached_id, ImageId merged_id) {
	auto& cached = cache.m_slot_images[cached_id];
	auto* merged = merged_id ? &cache.m_slot_images[merged_id] : nullptr;
	// PS5 raw block images can span several packed BC mips. Transfer their complete
	// guest bytes, including tail padding, before retiring the incompatible backing.
	if (cached.SafeToDownload()) {
		if (merged && merged->SafeToDownload() && !Publish(cache, *merged)) {
			return;
		}
		if (!Publish(cache, cached)) {
			return;
		}
		if (merged) {
			merged->MarkBufferModified();
		}
	}
	cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
	cache.FreeImage(cached_id);
}

} // namespace BcAlias

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto& cached = m_slot_images[cached_id];
	if (!cached.registered) {
		return {merged_id};
	}
	// PS5: image heaps keep BC images and raw block views of their layers resident together, so a
	// raw read of a layer subset with the same levels views the BC image instead of evicting it.
	const bool raw_view = binding == BindingType::Texture && !requested.IsBlock() &&
	                      requested.type == cached.info.type &&
	                      requested.data.address == cached.info.data.address &&
	                      requested.BlockExtent() == cached.info.BlockExtent() &&
	                      requested.resources.levels == cached.info.resources.levels &&
	                      requested.resources.layers < cached.info.resources.layers;
	if (!raw_view && BcAlias::Matches(requested, cached.info) && !cached.IsCpuDirty()) {
		BcAlias::Resolve(*this, cached_id, merged_id);
		return {merged_id};
	}
	const bool safe_to_delete = m_gc_tick - cached.tick_accessed_last > NumFramesBeforeRemoval &&
	                            cached.heap_refs.load(std::memory_order_relaxed) == 0u;

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     !std::equal(requested.mip_layout.begin(), requested.mip_layout.end(),
		                 cached.info.mip_layout.begin(),
		                 [](const ImageMipInfo& left, const ImageMipInfo& right) {
			                 return left.offset == right.offset && left.size == right.size &&
			                        left.pitch == right.pitch && left.height == right.height &&
			                        left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
			                        left.surface_z == right.surface_z;
		                 }))) {
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (PackedFloatMismatch(requested.pixel_format, cached.info.pixel_format)) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			return {ImageViewOps::FormatsCompatible(m_slot_images[result_id].info.pixel_format,
			                                        requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	return expanded_id;
}

bool TextureCache::CanDownload(const Image& image) const {
	const auto& info = image.info;
	if (info.samples != 1 || image.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		return IsSupportedDepthPlaneReadback(info) &&
		       Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
	}
	return info.metadata.compression == VideoOutCompression::Uncompressed &&
	       (!info.IsTiled() || !Prospero::IsFmaskTextureFormat(info.guest_format)) &&
	       info.LinearSize() != 0;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto& info        = image.info;
	const auto  binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	if (binding != BindingType::DepthTarget) {
		if ((binding == BindingType::RenderTarget || binding == BindingType::VideoOut) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		const auto linear = m_tiler.Detile(source.Handle(), source_offset, info);
		const auto copies = info.BufferCopies(linear.offset);
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
		return;
	}

	if (destination.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto layers          = info.resources.layers;
	const auto full_slice_size = info.linear_slice_stride;
	auto       linear          = m_tiler.Detile(source.Handle(), source_offset, info);
	auto       copies          = info.BufferCopies(linear.offset);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		copies.resize(1);
		copies[0].bufferOffset      = linear.offset;
		copies[0].bufferRowLength   = info.pitch;
		copies[0].bufferImageHeight = info.extent.height;
		copies[0].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, 0, layers};
		copies[0].imageExtent       = info.extent;
	}
	destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
}

// KYTY_UPLOAD_STATS=1: every 5 s, print texture upload bytes by cause, bytes whose guest contents
// hash equal to that image's previous upload (re-uploads of untouched data), the top images, and
// semaphore wait time per thread.
extern std::atomic<uint64_t> g_stats_wait_ns_gpu_thread, g_stats_wait_ns_other,
    g_stats_waits_gpu_thread, g_stats_waits_other;

namespace UploadStats {
struct Entry {
	uint64_t    size = 0, bytes = 0, uploads = 0, repeated = 0, last_hash = 0;
	uint32_t    mips = 0, layers = 0, tile = 0, width = 0, height = 0;
	vk::Format  format {};
	const char* cause = "";
};
static std::mutex                          g_mutex;
static std::unordered_map<uint64_t, Entry> g_images;
static uint64_t g_bytes_cpu = 0, g_bytes_maybe = 0, g_bytes_buffer = 0, g_bytes_repeated = 0;
static std::unordered_map<std::string, std::array<uint64_t, 2>> g_clear_calls; // checks, clears
static auto     g_last_print = std::chrono::steady_clock::now();

static bool Enabled() {
	static const bool enabled = std::getenv("KYTY_UPLOAD_STATS") != nullptr;
	return enabled;
}

static void Print(double seconds) {
	const double mb = 1024.0 * 1024.0;
	std::printf("[upload-stats] %.1fs: total %.0f MB/s | cpu-dirty %.0f | maybe-dirty %.0f | "
	            "buffer-modified %.0f | SAME-CONTENT %.0f MB/s\n",
	            seconds, (g_bytes_cpu + g_bytes_maybe + g_bytes_buffer) / mb / seconds,
	            g_bytes_cpu / mb / seconds, g_bytes_maybe / mb / seconds,
	            g_bytes_buffer / mb / seconds, g_bytes_repeated / mb / seconds);
	std::printf("[upload-stats] waits: gpu-thread %.0f ms/s (%.0f/s) | other threads %.0f ms/s "
	            "(%.0f/s)\n",
	            g_stats_wait_ns_gpu_thread.exchange(0) / 1e6 / seconds,
	            g_stats_waits_gpu_thread.exchange(0) / seconds,
	            g_stats_wait_ns_other.exchange(0) / 1e6 / seconds,
	            g_stats_waits_other.exchange(0) / seconds);
	std::vector<std::pair<uint64_t, Entry>> top(g_images.begin(), g_images.end());
	std::sort(top.begin(), top.end(),
	          [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
	for (size_t i = 0; i < std::min<size_t>(5, top.size()); i++) {
		const auto& [address, e] = top[i];
		if (e.bytes == 0) break;
		std::printf("[upload-stats]   #%zu 0x%010" PRIx64 " %ux%u %s mips=%u layers=%u tile=%u "
		            "size=%.1fMB uploads=%" PRIu64 " (%.0f MB/s, same-content %" PRIu64 ") %s\n",
		            i + 1, address, e.width, e.height, vk::to_string(e.format).c_str(), e.mips,
		            e.layers, e.tile, e.size / mb, e.uploads, e.bytes / mb / seconds, e.repeated,
		            e.cause);
	}
	std::vector<std::pair<std::string, std::array<uint64_t, 2>>> cl(g_clear_calls.begin(),
	                                                                g_clear_calls.end());
	std::sort(cl.begin(), cl.end(), [](const auto& a, const auto& b) { return a.second[1] > b.second[1]; });
	for (size_t i = 0; i < std::min<size_t>(6, cl.size()); i++) {
		std::printf("[clear-stats] %s: %.0f checks/s, %.0f clears/s\n", cl[i].first.c_str(),
		            cl[i].second[0] / seconds, cl[i].second[1] / seconds);
	}
	g_clear_calls.clear();
	std::fflush(stdout);
	for (auto& [address, e]: g_images) {
		e.bytes = 0; e.uploads = 0; e.repeated = 0;
	}
	g_bytes_cpu = g_bytes_maybe = g_bytes_buffer = g_bytes_repeated = 0;
}

static void Record(const Image& image) {
	const auto& info  = image.info;
	const char* cause = image.IsBufferModified()       ? "buffer-modified"
	                    : image.IsDefinitelyCpuDirty() ? "cpu-dirty"
	                                                   : "maybe-dirty";
	std::vector<uint8_t> bytes(info.data.size);
	uint64_t             hash = 0;
	if (LibKernel::Memory::TryReadBacking(info.data.address, bytes.data(), bytes.size())) {
		hash = XXH3_64bits(bytes.data(), bytes.size());
	}
	std::scoped_lock lock(g_mutex);
	auto&            e = g_images[info.data.address];
	const bool       same = hash != 0 && e.uploads + e.repeated > 0 && hash == e.last_hash;
	e.size = info.data.size; e.mips = info.resources.levels; e.layers = info.resources.layers;
	e.tile = static_cast<uint32_t>(info.tile_mode); e.width = info.extent.width;
	e.height = info.extent.height; e.format = info.pixel_format; e.cause = cause;
	e.bytes += info.data.size; e.uploads++; e.last_hash = hash;
	if (same) { e.repeated++; g_bytes_repeated += info.data.size; }
	(cause[0] == 'b' ? g_bytes_buffer : cause[0] == 'c' ? g_bytes_cpu : g_bytes_maybe) +=
	    info.data.size;
	const auto now     = std::chrono::steady_clock::now();
	const auto seconds = std::chrono::duration<double>(now - g_last_print).count();
	if (seconds >= 5.0) {
		Print(seconds);
		g_last_print = now;
	}
}
} // namespace UploadStats

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	TrackImage(id);
	// Images that cannot upload start with undefined host contents: AMD's Windows driver hands out
	// garbage, RADV mostly zeros. KYTY_ZERO_INIT=1 (experiment; did not fix DBZ's rainbow) clears them.
	static const bool zero_init = std::getenv("KYTY_ZERO_INIT") != nullptr;
	const bool        cannot_upload =
	    image.info.metadata.compression != VideoOutCompression::Uncompressed ||
	    image.info.samples > 1;
	if (cannot_upload && zero_init && !image.IsGpuModified()) {
		image.ZeroFillOnce();
	}
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		image.RefreshComplete();
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	// KYTY_SKIP_RT_REUPLOAD=1 (experiment): a render or depth target marked only by a buffer write
	// keeps its GPU contents instead of being re-copied from the buffer every frame.
	static const bool skip_rt_reupload = std::getenv("KYTY_SKIP_RT_REUPLOAD") != nullptr;
	if (upload && skip_rt_reupload && image.IsBufferModified() && !image.IsCpuDirty()) {
		const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
		if (binding == BindingType::RenderTarget || binding == BindingType::DepthTarget) {
			image.ClearBufferModified();
			image.RefreshComplete();
			return;
		}
	}
	if (upload) {
		ImageTrace::Log(image, "UPLOAD_FROM_MEMORY");
		if (UploadStats::Enabled()) {
			UploadStats::Record(image);
		}
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	}
	image.RefreshComplete();
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto  range = desc.info.metadata.range;
	const auto& view  = desc.view_info;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || view.level_count != 1) {
			return;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// Color metadata slices consist of complete 4 KiB blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	const bool volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto first          = volume_texture ? 0u : metadata_base_layer;
	const auto image_first    = volume_texture ? 0u : view.base_layer;
	const auto count =
	    volume_texture ? std::max(desc.info.extent.depth >> view.base_level, 1u) : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	const auto slice_size = range.size / layers;
	// Each slice holds the viewed mip's metadata in one run; DCC mips past the first tail mip
	// have none. FindImage moves base_level only for one-mip descriptors, which keep the slice.
	TileSizeOffset mip {.size = static_cast<uint32_t>(slice_size)};
	if (desc.info.metadata.kind == ImageMetadataKind::Dcc && desc.info.resources.levels > 1) {
		std::array<TileSizeOffset, 16> mips {};
		TileSizeAlign                  total {};
		(void)TileGetDccSize(desc.info.extent.width, desc.info.extent.height, layers,
		                     desc.info.bytes_per_block, desc.info.resources.levels,
		                     desc.info.tile_mode, total, 0, mips.data());
		mip = mips[view.base_level];
	}
	if (mip.size == 0) {
		return;
	}
	const GuestRange viewed {range.address + slice_size * first + mip.offset,
	                         slice_size * (count - 1) + mip.size};
	const auto run = [&](uint32_t slice) { return viewed.address + slice_size * slice; };
	// KYTY_SKIP_COLOR_META_CLEAR=1 (experiment): skip the per-key predicated clears that apply
	// GPU-written fast-clear metadata, to measure what they cost.
	static const bool skip_meta_clear = std::getenv("KYTY_SKIP_COLOR_META_CLEAR") != nullptr;
	if (skip_meta_clear && m_buffer_cache.IsRegionGpuModified(viewed.address, viewed.size)) {
		return;
	}
	if (m_buffer_cache.IsRegionGpuModified(viewed.address, viewed.size)) {
		std::array<vk::ClearAttachment, TileManager::ColorClearSlots> clears {};
		uint32_t keys = 0;
		for (uint32_t key = 0; key < clears.size(); key++) {
			clears[key].aspectMask = vk::ImageAspectFlagBits::eColor;
			if (DecodeColorClear(desc, static_cast<uint8_t>(key << 5u),
			                     clears[key].clearValue.color)) {
				keys |= 1u << key;
			}
		}
		// No selected key to clear, or none left since the last check.
		bool checked = true;
		for (uint32_t slice = 0; slice < count && checked; slice++) {
			checked = m_buffer_cache.IsColorMetadataChecked(run(slice), mip.size, keys);
		}
		if (checked) {
			return;
		}
		{
			std::scoped_lock lock {m_lock};
			auto&            image = m_slot_images[id];
			// Compressed images cannot upload, so they must already hold native contents.
			const bool native_current = (image.usage.render_target || image.IsGpuModified()) &&
			                            !image.IsBufferModified() && !image.IsCpuDirty();
			// 2D-array attachment views select 2D layers or, of a volume, depth slices.
			if (m_graphics.conditional_rendering_enabled &&
			    count <= m_graphics.GetPhysicalDeviceProperties().limits.maxFramebufferLayers &&
			    (image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment) &&
			    (m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
			     vk::FormatFeatureFlagBits::eColorAttachment) &&
			    (desc.info.metadata.compression == VideoOutCompression::Uncompressed ||
			     native_current)) {
				// The GPU selects each slice's clear, so the CPU never waits for the metadata.
				Common::DebugCounters::Add(Common::DebugCounters::Counter::ColorMetadataChecks);
				InitializeImage(id);
				const bool expand = desc.type != BindingType::VideoOut;
				const auto [buffer, offset] =
				    m_buffer_cache.ObtainBuffer(viewed.address, viewed.size, expand);
				const auto predicates = m_tiler.CheckColorClear(
				    {buffer->Handle(), offset, viewed.size}, slice_size, count, keys, expand);
				for (uint32_t slice = 0; slice < count && expand; slice++) {
					m_buffer_cache.MarkColorMetadataChecked(run(slice), mip.size, keys);
				}
				if (UploadStats::Enabled()) {
					std::scoped_lock stats_lock(UploadStats::g_mutex);
					char key_text[160];
					std::snprintf(key_text, sizeof(key_text), "%ux%u %s view=%s fmt=%s count=%u keys=%u",
					              desc.info.extent.width, desc.info.extent.height,
					              volume_texture ? "3D" : "2D", vk::to_string(view.type).c_str(),
					              vk::to_string(view.format).c_str(), count, std::popcount(keys));
					auto& c = UploadStats::g_clear_calls[key_text];
					c[0] += 1;
					c[1] += static_cast<uint64_t>(count) * std::popcount(keys);
				}
				// Many-slice targets: one compute dispatch writes each slice's hit key instead of
				// count * keys predicated clears (64-slice targets cost ~320 clears per check).
				static const bool no_fast_apply = std::getenv("KYTY_NO_FAST_CLEAR_APPLY") != nullptr;
				const auto numeric = vk::componentNumericFormat(view.format, 0);
				const bool float_format = std::strcmp(numeric, "UINT") != 0 &&
				                          std::strcmp(numeric, "SINT") != 0;
				if (!no_fast_apply && count >= 8 && float_format &&
				    (image.backing.usage & vk::ImageUsageFlagBits::eStorage) &&
				    (m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
				     vk::FormatFeatureFlagBits::eStorageImage)) {
					// A 3D image is always written through a 3D view: a 2D-array view of a 3D image
					// is valid only as an attachment, not as a storage image. AMD's Windows driver
					// tiles 3D images differently and scrambles them when written that way (DBZ's
					// 64^3 color-grading LUT, viewed as 64 layers, came out as a rainbow). Slices of
					// such a 2D-array view are depth slices, starting at its base layer.
					const bool image_3d = desc.info.IsVolume();
					ImageViewInfo storage {};
					storage.format      = view.format;
					storage.type        = image_3d ? vk::ImageViewType::e3D : vk::ImageViewType::e2DArray;
					storage.base_level  = view.base_level;
					storage.base_layer  = image_3d ? 0u : image_first;
					storage.layer_count = image_3d ? 1u : count;
					storage.usage       = vk::ImageUsageFlagBits::eStorage;
					m_scheduler.EndRendering();
					image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderStorageWrite,
					              {}, m_scheduler.Current().Handle());
					std::array<vk::ClearColorValue, TileManager::ColorClearSlots> colors {};
					for (uint32_t key = 0; key < colors.size(); key++) {
						colors[key] = clears[key].clearValue.color;
					}
					m_tiler.ApplyColorClear(predicates, image.FindView(storage), image_3d,
					                        std::max(image.info.extent.width >> view.base_level, 1u),
					                        std::max(image.info.extent.height >> view.base_level, 1u),
					                        count, image_3d ? image_first : 0u, keys, colors);
					CommitGpuWrite(image);
					return;
				}
				const auto command = m_scheduler.Current().Handle();
				const auto area    = BeginColorRendering(image, view.format, view.base_level,
				                                         image_first, count, nullptr, command);
				vk::ClearRect rect {.rect = area, .layerCount = 1};
				for (uint32_t slice = 0; slice < count; slice++) {
					rect.baseArrayLayer = slice;
					for (uint32_t key = 0; key < clears.size(); key++) {
						if ((keys & (1u << key)) != 0) {
							const vk::ConditionalRenderingBeginInfoEXT predicate {
							    .buffer = predicates,
							    .offset = (slice * clears.size() + key) * sizeof(uint32_t)};
							command.beginConditionalRenderingEXT(&predicate);
							command.clearAttachments(1, &clears[key], 1, &rect);
							command.endConditionalRenderingEXT();
						}
					}
				}
				command.endRendering();
				CommitGpuWrite(image);
				return;
			}
		}
		// Finish native metadata writes before reading backing bytes. This can submit the
		// scheduler, so discovery runs before final draw uploads and never holds the texture
		// lock across it.
		Common::DebugCounters::ScopedTimer timer(
		    Common::DebugCounters::Counter::ColorMetadataReadbacks,
		    Common::DebugCounters::Counter::ColorMetadataReadbackNs);
		m_buffer_cache.ReadMemory(viewed.address, viewed.size, false);
	}
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = run(slice);
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		std::vector<uint8_t> bytes(mip.size);
		if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			EXIT("TextureCache: failed to read color metadata slice\n");
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			continue;
		}
		{
			std::scoped_lock lock {m_lock};
			const auto& image = m_slot_images[id];
			const bool depth = image.info.IsDepth();
			if (depth) {
				const auto value = clear.color.float32[0];
				clear.depthStencil = {value, 0u};
			}
			ClearImage(m_scheduler.Current(), id, depth ? image.backing.format : view.format,
			           {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
			            view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, bytes.size());
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
	}
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		image.ResolveMaybeCpuHash(image.HashGuestEdges());
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id);
}

ImageId TextureCache::FindStencilPlane(const Image& depth, GuestRange range) const {
	for (const auto id: depth.stencil_ids) {
		if (m_slot_images[id].info.data == range) {
			return id;
		}
	}
	return {};
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association = FindStencilPlane(depth, stencil);
	if (!association) {
		ForEachImageWithAddress(stencil.address, [&](ImageId id, Image& owner) {
			if (owner.info.data == stencil && owner.info.extent == depth.info.extent) {
				association = id;
			}
		});
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		ImageInfo plane {};
		plane.data         = stencil;
		plane.pixel_format = vk::Format::eS8Uint;
		plane.guest_format = Prospero::BufferFormat::k8UInt;
		plane.type         = depth.info.type;
		plane.extent       = depth.info.extent;
		plane.resources    = {depth.stencil_subresources.level_count,
		                      depth.stencil_subresources.layer_count};
		plane.tile_mode    = depth.info.tile_mode;
		plane.pitch =
		    plane.IsTiled() ? TileGetRenderTargetPitch(plane.extent.width, 1) : depth.info.pitch;
		plane.bytes_per_block = 1;
		plane.UpdateSize();
		if (record.depth_id != depth_id) {
			if (record.depth_id) {
				std::erase(m_slot_images[record.depth_id].stencil_ids, association);
			}
			depth.stencil_ids.push_back(association);
		}
		record.info                 = plane;
		record.depth_id             = depth_id;
		record.stencil_subresources = depth.stencil_subresources;
	}

	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	Common::PerfStats::ScopedNs perf_timer(Common::PerfStats::g_ns_find_image);
	ValidateImageDesc(desc);
	EXIT_IF(desc.info.data.Empty());
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		Common::SmallVector<ImageId, 8> candidates;
		ForEachImageInRegion(desc.info.data.address, desc.info.data.size,
		                     [&](ImageId id, Image&) { candidates.push_back(id); });
		bool block_alias = false;

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			// Stencil descriptors name the associated plane; its depth owner supplies
			// the native image and the canonical guest tiling.
			if ((desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
			    image.depth_id && image.info.data == desc.info.data &&
			    m_slot_images[image.depth_id].info.samples == desc.info.samples &&
			    image.info.extent == desc.info.extent && image.info.type == desc.info.type &&
			    image.info.resources.levels >= desc.info.resources.levels &&
			    image.info.resources.layers >= desc.info.resources.layers &&
			    desc.info.bytes_per_block == 1 &&
			    desc.info.guest_format == Prospero::BufferFormat::k8UInt) {
				result = id;
				break;
			}
			if (SameBacking(image.info, desc.info, exact_format)) {
				result = id;
			} else if (image.SafeToDownload() &&
			           BcAlias::Matches(desc.info, image.info)) {
				block_alias = true;
			}
		}

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
			// PS5: image heaps hold a descriptor per dynamic-resolution extent of a render target,
			// and only one extent at an address holds data, so heap lookups create no second one.
			if (!result && desc.heap &&
			    std::any_of(candidates.begin(), candidates.end(), [&](ImageId id) {
				    const auto* image = m_slot_images.try_get(id);
				    return image != nullptr && image->registered &&
					       image->heap_refs.load(std::memory_order_relaxed) != 0u &&
					       image->info.data.address == desc.info.data.address &&
					       image->info.type == desc.info.type &&
					       image->info.extent != desc.info.extent;
			    })) {
				return {};
			}
		} else if (block_alias) {
			for (const auto candidate: candidates) {
				const auto& owner = m_slot_images[candidate];
				if (candidate != result && owner.registered && !owner.IsCpuDirty() &&
				    BcAlias::Matches(m_slot_images[result].info, owner.info)) {
					BcAlias::Resolve(*this, candidate, result);
				}
			}
		}

		if (result && !m_slot_images[result].depth_id) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result);
				result = {};
			}
		}
		if (!result) {
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
			// PPSA15210 (UE transient pool): a float image first found as a storage target is a
			// fresh render-graph texture whose contents are undefined. Uploading the stale guest
			// bytes of whatever used the memory before read back as NaN/Inf packed floats; when
			// tile classification then skipped every tile, the bloom spread them (the rainbow).
			// Start it at zero instead. KYTY_UPLOAD_NEW_STORAGE=1 restores the upload.
			static const bool upload_new_storage = std::getenv("KYTY_UPLOAD_NEW_STORAGE") != nullptr;
			const auto        numeric = vk::componentNumericFormat(inserted.info.pixel_format, 0);
			const bool        float_format =
			    std::strcmp(numeric, "SFLOAT") == 0 || std::strcmp(numeric, "UFLOAT") == 0;
			// Bloom render targets reuse the same pool and accumulate with additive blending.
			const bool fresh_target =
			    desc.type == BindingType::Storage || desc.type == BindingType::RenderTarget;
			if (!upload_new_storage && fresh_target && float_format &&
			    inserted.info.samples == 1 && !inserted.info.IsDepth() &&
			    inserted.info.metadata.compression == VideoOutCompression::Uncompressed) {
				TrackImage(result);
				inserted.ClearBufferModified();
				inserted.RefreshComplete();
				inserted.ZeroFillOnce();
				ImageTrace::Log(inserted, "zero_filled_new_storage");
			}
			ImageTrace::Log(inserted, "new", desc.type == BindingType::Storage ? "for=storage"
			                                 : desc.type == BindingType::RenderTarget
			                                     ? "for=rt"
			                                     : "for=other");
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_gc_tick;
		TouchImage(image);
	}
	if (ImageTrace::Get().enabled && desc.info.extent.width == ImageTrace::Get().width &&
	    desc.info.extent.height == ImageTrace::Get().height &&
	    ImageTrace::g_flip.load(std::memory_order_relaxed) >= ImageTrace::Get().from) {
		const auto& found = m_slot_images[result];
		const auto groups = ImageTrace::g_cs_groups.load(std::memory_order_relaxed);
		std::printf("TRACE f=%" PRIu64 " CS=%016" PRIx64 " groups=%ux%ux%u%s\n",
		            ImageTrace::g_flip.load(std::memory_order_relaxed),
		            ImageTrace::g_cs_hash.load(std::memory_order_relaxed),
		            static_cast<uint32_t>(groups & 0x1FFFFF),
		            static_cast<uint32_t>((groups >> 21) & 0x1FFFFF),
		            static_cast<uint32_t>((groups >> 42) & 0x1FFFFF),
		            (groups >> 63) != 0 ? " INDIRECT" : "");
		std::printf("TRACE f=%" PRIu64 " FIND %s addr=%" PRIx64 " want_fmt%u guest%u tile%u view_fmt%u"
		            " meta%u -> s%06" PRIu64 " fmt%u\n",
		            ImageTrace::g_flip.load(std::memory_order_relaxed),
		            desc.type == BindingType::Storage        ? "storage"
		            : desc.type == BindingType::RenderTarget ? "rt"
		            : desc.type == BindingType::Texture      ? "texture"
		                                                     : "other",
		            desc.info.data.address, static_cast<uint32_t>(desc.info.pixel_format),
		            static_cast<uint32_t>(desc.info.guest_format),
		            static_cast<uint32_t>(desc.info.tile_mode),
		            static_cast<uint32_t>(desc.view_info.format),
		            static_cast<uint32_t>(desc.info.metadata.kind), found.serial,
		            static_cast<uint32_t>(found.info.pixel_format));
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	std::scoped_lock lock {m_lock};
	Common::SmallVector<ImageId, 4> matches;
	ForEachImageWithAddress(address, [&](ImageId id, Image& image) {
		const auto* owner = &image;
		if (ensure_valid && owner->depth_id) {
			owner = &m_slot_images[owner->depth_id];
		}
		if (ensure_valid && !owner->SafeToDownload()) {
			return;
		}
		matches.push_back(id);
	});
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid && m_slot_images[selected].depth_id) {
		selected = m_slot_images[selected].depth_id;
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
	}
	// Storage readback also queues the first GPU write since the CPU last wrote the image: an
	// image rewritten every frame would otherwise be downloaded at every GL2 writeback. GPU data
	// in the buffer cache over the whole image is not a CPU write. Read before RefreshImage
	// uploads the CPU write.
	const bool first_write =
	    m_readback_storage_images && (!image.IsGpuModified() || image.IsDefinitelyCpuDirty()) &&
	    !m_buffer_cache.HasOnlyGpuDirtyBytes(image.info.data.address, image.info.data.size);
	ImageTrace::Log(image, desc.type == BindingType::Storage ? "bind_storage" : "bind_texture");
	RefreshImage(id);
	if (image.info.HasStencil() &&
	    desc.info.data.address >= image.info.stencil.address &&
	    desc.info.data.End() <= image.info.stencil.End()) {
		if (const auto stencil_id = FindStencilPlane(image, image.info.stencil)) {
			RefreshImage(stencil_id);
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			CommitGpuWrite(image);
			TrackImageDownload(id, image);
			if (first_write) {
				m_download_images.insert(id);
			}
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	ImageTrace::Log(image, "bind_rt");
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const auto [metadata, inserted] = m_surface_metas.try_emplace(desc.info.metadata.range.address);
		if (inserted && image.info.htile_initial_clear) {
			metadata->second.clear_mask.set();
		}
		metadata->second.owner = id;
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil));
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	image.RefreshComplete();
	image.MarkGpuModified();
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	Common::SmallVector<ImageId, 8> image_ids;
	ForEachImageInRegion(address, size, [&](ImageId id, Image&) { image_ids.push_back(id); });
	for (const auto id: image_ids) {
		const auto&          owner = m_slot_images[id];
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner.info.resources.levels, 0, owner.info.TransferLayers()};
		if (owner.depth_id && owner.info.data.address == address &&
		    owner.info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner.depth_id;
			candidate_subresources = owner.stencil_subresources;
			const auto& depth      = m_slot_images[candidate_id];
			if (depth.backing.image == nullptr || !depth.info.HasStencil()) {
				continue;
			}
		} else if (!owner.depth_id && owner.info.data.address == address &&
		           owner.info.data.size == size) {
			candidate = owner.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                 : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		BeginColorRendering(image, format, range.baseMipLevel, range.baseArrayLayer,
		                    range.layerCount, &clear, command.Handle());
		command.Handle().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

vk::Rect2D TextureCache::BeginColorRendering(Image& image, vk::Format format, uint32_t level,
                                             uint32_t layer, uint32_t layers,
                                             const vk::ClearValue* clear,
                                             vk::CommandBuffer     command) {
	ImageViewInfo view {};
	view.format      = format;
	view.type        = layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	view.base_level  = level;
	view.base_layer  = layer;
	view.layer_count = layers;
	view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	              vk::AccessFlagBits2::eColorAttachmentRead |
	                  vk::AccessFlagBits2::eColorAttachmentWrite,
	              {}, command);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = image.FindView(view);
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	if (clear != nullptr) {
		attachment.loadOp     = vk::AttachmentLoadOp::eClear;
		attachment.clearValue = *clear;
	}
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {std::max(image.info.extent.width >> level, 1u),
	                                  std::max(image.info.extent.height >> level, 1u)};
	rendering.layerCount           = layers;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	return rendering.renderArea;
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.linear_slice_stride;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			const auto copies = info.BufferCopies(destination_offset);
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
		} else {
			const auto copies = info.BufferCopies();
			m_tiler.TileImage(image, copies, destination.Handle(), destination_offset,
			                  info.data.size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	const std::array copies {vk::BufferImageCopy {0,
	                                              info.pitch,
	                                              info.extent.height,
	                                              {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                                              {},
	                                              info.extent}};
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto guest_linear = tiled ? m_tiler.GetScratchBuffer(info.LinearSize(), host_linear.buffer)
	                          : TileManager::Result {destination.Handle(), destination_offset,
	                                                 destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (tiled) {
		m_tiler.Tile(guest_linear.buffer, guest_linear.offset, guest_linear.size,
		             destination.Handle(), destination_offset, info.data.size, info);
	}
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                 uint64_t destination_size, uint32_t levels) {
	if (UploadBinding(image) == BindingType::DepthTarget) {
		EXIT_IF(destination_size != image.info.data.size || levels != image.info.resources.levels);
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	const auto& info      = image.info;
	const auto  transform = info.GetColorTransform();
	if (info.IsTiled()) {
		const auto copies = info.BufferCopies(0, levels);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, destination_size,
		                  levels);
	} else if (transform != ColorTransform::None) {
		auto       linear = m_tiler.GetScratchBuffer(destination_size);
		const auto copies = info.BufferCopies(0, levels);
		image.Download(copies, linear.buffer, 0, linear.size);
		m_tiler.TransformColor(linear, {destination.Handle(), destination_offset, destination_size},
		                       transform, false);
	} else {
		const auto copies = info.BufferCopies(destination_offset, levels);
		image.Download(copies, destination.Handle(), destination_offset, destination_size);
	}
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	if (!m_texture_cache.CanDownload(image) ||
	    (TextureCache::UploadBinding(image) == TextureCache::BindingType::DepthTarget &&
	     copy_size != image.info.data.size)) {
		return false;
	}
	ForgetColorMetadataChecks(image.info.data.address, copy_size);
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, levels);
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	if (!image.SafeToDownload() || !CanDownload(image)) {
		return false;
	}
	ImageTrace::Log(image, "download_to_memory");
	const auto range = image.info.data;
	auto download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download)
	                    .MapDownload(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	if (!LibKernel::Memory::TryReadBacking(range.address, download.mapped, range.size)) {
		return false;
	}
	download.buffer->Flush(download.offset, range.size);

	DownloadImage(image, *download.buffer, download.offset, range.size,
	              image.info.resources.levels);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureDownloadBytes, range.size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.buffer->Handle();
	barrier.offset              = download.offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	// With storage readback, guest accesses to its pages wait until the stored data lands.
	if (m_readback_storage_images) {
		{
			std::scoped_lock write_back_lock {m_write_back_mutex};
			m_write_backs.push_back({range, m_scheduler.CurrentTick()});
		}
		m_page_manager.UpdatePageWatchers<true, true>(range.address, range.size);
	}
	m_scheduler.DeferPriorityOperation([this, download = std::move(download), range,
	                                    trapped = m_readback_storage_images] {
		download.buffer->Invalidate(download.offset, range.size);
		LibKernel::Memory::WriteBacking(range.address, download.mapped, range.size);
		if (trapped) {
			m_page_manager.UpdatePageWatchers<false, true>(range.address, range.size);
			// Priority operations run in queue order, so this write-back is the oldest.
			std::scoped_lock write_back_lock {m_write_back_mutex};
			m_write_backs.erase(m_write_backs.begin());
		}
	});
	return true;
}

// Diagnostic: with KYTY_PERF_STATS=1, every 60 flips print FPS, GPU waits and
// cache sizes to stdout, to see what grows over time and what stalls the GPU thread.
void TextureCache::PrintPerfStats(uint64_t flip) {
	namespace DC = Common::DebugCounters;
	namespace MS = Common::PerfStats;
	static const bool enabled = std::getenv("KYTY_PERF_STATS") != nullptr;
	constexpr uint64_t Window = 60;
	if (!enabled) {
		return;
	}
	struct Snapshot {
		std::chrono::steady_clock::time_point time {};
		uint64_t gpu_waits = 0, gpu_wait_ns = 0, meta_readbacks = 0, meta_ns = 0, submits = 0;
		uint64_t draws = 0, dispatches = 0, evictions = 0, shaders = 0, lds_avoided = 0;
		uint64_t sync_downloads = 0, sync_ns = 0, yield_sleeps = 0;
		uint64_t gpu_thread_waits = 0, gpu_thread_wait_ns = 0, gpu_thread_busy_ns = 0;
		uint64_t rb_fault_read = 0, rb_fault_write = 0, rb_gpu_thread = 0;
		uint64_t ns_draw = 0, ns_dispatch = 0, ns_find_image = 0, ns_pipeline = 0;
	};
	static Snapshot previous {};
	DC::SetEnabled(true); // the panel's counters only count while enabled
	if (flip % Window != 0) {
		return;
	}
	Snapshot now {};
	now.time           = std::chrono::steady_clock::now();
	now.gpu_waits      = DC::Get(DC::Counter::GpuWaits);
	now.gpu_wait_ns    = DC::Get(DC::Counter::GpuWaitNs);
	now.meta_readbacks = DC::Get(DC::Counter::ColorMetadataReadbacks);
	now.meta_ns        = DC::Get(DC::Counter::ColorMetadataReadbackNs);
	now.submits        = DC::Get(DC::Counter::QueueSubmits);
	now.draws          = DC::Get(DC::Counter::Draws);
	now.dispatches     = DC::Get(DC::Counter::Dispatches);
	now.evictions      = DC::Get(DC::Counter::TextureEvictions);
	now.shaders        = DC::Get(DC::Counter::ShadersCompiled);
	now.lds_avoided    = MS::g_lds_stalls_avoided.load(std::memory_order_relaxed);
	now.sync_downloads = MS::g_sync_downloads.load(std::memory_order_relaxed);
	now.sync_ns        = MS::g_sync_wait_ns.load(std::memory_order_relaxed);
	now.yield_sleeps   = MS::g_yield_sleeps.load(std::memory_order_relaxed);
	now.gpu_thread_waits   = g_stats_waits_gpu_thread.load(std::memory_order_relaxed);
	now.gpu_thread_wait_ns = g_stats_wait_ns_gpu_thread.load(std::memory_order_relaxed);
	now.gpu_thread_busy_ns = DC::Get(DC::Counter::GpuThreadBusyNs);
	now.rb_fault_read      = MS::g_rb_fault_read.load(std::memory_order_relaxed);
	now.rb_fault_write     = MS::g_rb_fault_write.load(std::memory_order_relaxed);
	now.rb_gpu_thread      = MS::g_rb_gpu_thread.load(std::memory_order_relaxed);
	now.ns_draw            = MS::g_ns_draw.load(std::memory_order_relaxed);
	now.ns_dispatch        = MS::g_ns_dispatch.load(std::memory_order_relaxed);
	now.ns_find_image      = MS::g_ns_find_image.load(std::memory_order_relaxed);
	now.ns_pipeline        = MS::g_ns_pipeline.load(std::memory_order_relaxed);
	if (previous.time != std::chrono::steady_clock::time_point {}) {
		const double seconds = std::chrono::duration<double>(now.time - previous.time).count();
		size_t       images  = 0;
		m_slot_images.ForEach([&](ImageId, const Image&) { ++images; });
		const auto per_frame = [&](uint64_t a, uint64_t b) {
			return static_cast<double>(a - b) / static_cast<double>(Window);
		};
		const auto ms = [](uint64_t a, uint64_t b) { return static_cast<double>(a - b) / 1e6; };
		std::printf("PERF f=%" PRIu64 " fps=%.1f | per frame: draws=%.0f disp=%.0f submits=%.1f "
		            "gpuWaits=%.1f (%.0f ms) drawThread=%.1f (%.0f ms, busy %.0f ms) syncReads=%.2f (%.0f ms) [faultRead=%.2f faultWrite=%.2f threadReads=%.2f] metaReads=%.2f (%.0f ms) "
		            "ldsAvoided=%.1f | 60 frames: shaders=%" PRIu64 " evictions=%" PRIu64
		            " sleeps=%" PRIu64 " | textures=%zu memMB=%" PRIu64 " gcFromMB=%" PRIu64
		            " metas=%zu | ms en 60 frames: dibujos=%.0f disp=%.0f buscarTex=%.0f programas=%.0f\n",
		            flip, static_cast<double>(Window) / seconds, per_frame(now.draws, previous.draws),
		            per_frame(now.dispatches, previous.dispatches),
		            per_frame(now.submits, previous.submits),
		            per_frame(now.gpu_waits, previous.gpu_waits),
		            ms(now.gpu_wait_ns, previous.gpu_wait_ns),
		            per_frame(now.gpu_thread_waits, previous.gpu_thread_waits),
		            ms(now.gpu_thread_wait_ns, previous.gpu_thread_wait_ns),
		            ms(now.gpu_thread_busy_ns, previous.gpu_thread_busy_ns),
		            per_frame(now.sync_downloads, previous.sync_downloads),
		            ms(now.sync_ns, previous.sync_ns),
		            per_frame(now.rb_fault_read, previous.rb_fault_read),
		            per_frame(now.rb_fault_write, previous.rb_fault_write),
		            per_frame(now.rb_gpu_thread, previous.rb_gpu_thread),
		            per_frame(now.meta_readbacks, previous.meta_readbacks),
		            ms(now.meta_ns, previous.meta_ns),
		            per_frame(now.lds_avoided, previous.lds_avoided), now.shaders - previous.shaders,
		            now.evictions - previous.evictions, now.yield_sleeps - previous.yield_sleeps,
		            images, m_total_used_memory >> 20u, m_trigger_gc_memory >> 20u,
		            m_surface_metas.size(), ms(now.ns_draw, previous.ns_draw),
		            ms(now.ns_dispatch, previous.ns_dispatch),
		            ms(now.ns_find_image, previous.ns_find_image),
		            ms(now.ns_pipeline, previous.ns_pipeline));
		if (flip % (Window * 5) == 0) {
			std::printf("PERF_PAGES f=%" PRIu64 " stale=%" PRIu64 " prefetched=%" PRIu64
			            " hits=%" PRIu64 "\n  syncDownloads:%s\n  writeFaults:%s\n  readFaults:%s\n",
			            flip, MS::g_stale_reads.load(std::memory_order_relaxed),
			            MS::g_prefetches.load(std::memory_order_relaxed),
			            MS::g_prefetch_hits.load(std::memory_order_relaxed),
			            MS::g_pages_sync_download.TakeTop(8).c_str(),
			            MS::g_pages_fault_write.TakeTop(8).c_str(),
			            MS::g_pages_fault_read.TakeTop(8).c_str());
		}
		std::fflush(stdout);
	}
	previous = now;
}

// Diagnostic only: with KYTY_DUMP_DIR set, the flip after `<dir>\trigger` appears copies mip 0 of
// every GPU-written or volume color image to `<dir>\f<flip>_...bin` and deletes the trigger.
void TextureCache::DumpImagesIfRequested(uint64_t flip) {
	ImageTrace::g_flip.store(flip, std::memory_order_relaxed);
	PrintPerfStats(flip);
	static const char* dir = std::getenv("KYTY_DUMP_DIR");
	if (dir == nullptr) {
		return;
	}
	const std::string trigger = std::string(dir) + "\\trigger";
	if (std::FILE* probe = std::fopen(trigger.c_str(), "rb"); probe != nullptr) {
		std::fclose(probe);
	} else {
		return;
	}
	std::remove(trigger.c_str());
	ImageTrace::g_buffer_dump_budget.store(16, std::memory_order_relaxed);

	static constexpr uint64_t DUMP_BUDGET = 3ull << 30;
	uint64_t                  total       = 0;
	uint32_t                  count       = 0;
	std::scoped_lock          lock {m_lock};
	m_slot_images.ForEach([&](ImageId, Image& image) {
		const auto& info = image.info;
		if (image.backing.image == nullptr || info.IsDepth() || info.samples != 1 ||
		    image.depth_id || vk::isCompressed(info.pixel_format) ||
		    info.pixel_format == vk::Format::eUndefined) {
			return;
		}
		if (!image.IsGpuModified() && !info.IsVolume()) {
			return;
		}
		const uint32_t bpp    = vk::blockSize(info.pixel_format);
		const uint32_t layers = info.IsVolume() ? 1u : std::max(1u, info.resources.layers);
		const uint64_t size   = uint64_t {bpp} * info.extent.width * info.extent.height *
		                      info.extent.depth * layers;
		if (bpp == 0 || size == 0 || total + size > DUMP_BUDGET) {
			return;
		}
		total += size;
		count++;
		vk::BufferImageCopy copy {};
		copy.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		copy.imageSubresource.mipLevel       = 0;
		copy.imageSubresource.baseArrayLayer = 0;
		copy.imageSubresource.layerCount     = layers;
		copy.imageExtent                     = info.extent;
		auto download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download)
		                    .MapDownload(size, std::max<uint32_t>(bpp, 4));
		copy.bufferOffset = download.offset;
		image.Download(std::span(&copy, 1), download.buffer->Handle(), download.offset, size);
		vk::BufferMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = download.buffer->Handle();
		barrier.offset              = download.offset;
		barrier.size                = size;
		m_scheduler.EndRendering();
		m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                                               vk::PipelineStageFlagBits::eHost, {}, 0,
		                                               nullptr, 1, &barrier, 0, nullptr);
		char name[512];
		std::snprintf(name, sizeof(name),
		              "%s\\f%05" PRIu64 "_s%06" PRIu64 "_%ux%ux%u_L%u_fmt%u_bpp%u_g%u_%s_%s_%" PRIx64
		              ".bin",
		              dir, flip, image.serial, info.extent.width, info.extent.height,
		              info.extent.depth, layers, static_cast<uint32_t>(info.pixel_format), bpp,
		              static_cast<uint32_t>(info.guest_format), info.IsVolume() ? "3d" : "2d",
		              image.usage.render_target ? "rt" : (image.IsGpuModified() ? "gpu" : "tex"),
		              info.data.address);
		m_scheduler.DeferPriorityOperation(
		    [download = std::move(download), size, path = std::string(name)] {
			    download.buffer->Invalidate(download.offset, size);
			    if (std::FILE* out = std::fopen(path.c_str(), "wb"); out != nullptr) {
				    std::fwrite(download.mapped, 1, static_cast<size_t>(size), out);
				    std::fclose(out);
			    }
		    });
	});
	std::printf("KYTY_DUMP flip=%" PRIu64 " images=%u bytes=%" PRIu64 "\n", flip, count, total);
	std::fflush(stdout);
}

uint64_t TextureCache::WriteBackTick(uint64_t address, uint64_t size) {
	if (!m_readback_storage_images) {
		return 0;
	}
	const auto begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	uint64_t   tick  = 0;

	std::scoped_lock write_back_lock {m_write_back_mutex};
	for (const auto& write_back: m_write_backs) {
		if (write_back.range.address < end && begin < write_back.range.End()) {
			tick = std::max(tick, write_back.tick);
		}
	}
	return tick;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	ForEachImageInRegion(address, size, [&](ImageId, Image& image) {
		if (image.info.data.address != address) {
			return;
		}
		ImageTrace::Log(image, "GPU_BUFFER_WRITE_INVALIDATES");
		image.ClearGpuModified();
		image.MarkBufferModified();
	});
}

bool TextureCache::CopyToImage(uint64_t address, uint64_t source, uint64_t size) {
	std::scoped_lock lock {m_lock};
	Common::SmallVector<ImageId, 8> targets;
	ForEachImageInRegion(address, size, [&](ImageId id, Image&) { targets.push_back(id); });
	if (targets.size() != 1) {
		return false;
	}
	auto&       image           = m_slot_images[targets.front()];
	const auto& info            = image.info;
	bool        source_modified = false;
	ForEachImageInRegion(source, size, [&](ImageId, Image& owner) {
		source_modified = owner.IsGpuModified();
		return source_modified;
	});
	if (image.depth_id || !image.SafeToDownload() || !CanDownload(image) || !info.IsTiled() ||
	    info.IsVolume() || info.IsDepth() || info.HasMetadata() || address < info.data.address ||
	    source_modified) {
		return false;
	}
	// Each layer holds its levels from the smallest up, and the tail levels share one block.
	const auto layers  = info.resources.layers;
	const auto stride  = info.data.size / layers;
	const auto offset  = address - info.data.address;
	const auto begin   = offset % stride;
	const auto end     = begin + size;
	uint64_t   covered = begin;
	uint32_t   first   = 0;
	uint32_t   count   = 0;
	for (uint32_t level = info.resources.levels; level-- > 0;) {
		const auto& mip     = info.mip_layout[level];
		const auto  mip_end = mip.offset + mip.size / layers;
		if (mip_end <= begin || mip.offset >= end) {
			continue;
		}
		if (mip.offset < begin) {
			return false;
		}
		covered = mip_end;
		first   = level;
		count++;
	}
	if (covered != end) {
		return false;
	}
	ImageInfo  part   = info;
	const auto extent = info.MipExtent(first);
	part.data.size    = size;
	part.extent       = {extent.width, extent.height, 1};
	part.resources    = {count, 1};
	part.first_tail_level -= std::min(first, info.first_tail_level);
	uint64_t linear = 0;
	for (uint32_t level = 0; level < count; level++) {
		auto& mip = part.mip_layout[level] = info.mip_layout[first + level];
		mip.offset -= begin;
		mip.size /= layers;
		mip.linear_offset = Common::AlignUp<uint64_t>(linear, std::max(info.bytes_per_block, 4u));
		linear            = mip.linear_offset + mip.linear_size;
	}
	const auto [buffer, source_offset] = m_buffer_cache.ObtainBuffer(source, size, false);
	const auto texels = m_tiler.Detile(buffer->Handle(), source_offset, part);
	auto       copies = part.BufferCopies(texels.offset);
	for (auto& copy: copies) {
		copy.imageSubresource.mipLevel += first;
		copy.imageSubresource.baseArrayLayer = static_cast<uint32_t>(offset / stride);
	}
	image.Upload(copies, texels.buffer, texels.offset, texels.size);
	return true;
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	const auto       page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto       page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	Common::SmallVector<ImageId, 8> image_ids;
	ForEachImageInRegion(page_begin, page_end - page_begin,
	                     [&](ImageId id, Image&) { image_ids.push_back(id); });
	for (const auto id: image_ids) {
		auto& owner = m_slot_images[id];
		if (owner.Overlaps(address, size)) {
			ImageTrace::Log(owner, "CPU_WRITE_DIRTY");
			owner.MarkCpuDirty();
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner.info.data.address;
		const auto image_end   = owner.info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	return found->second.clear_mask.test(slice);
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask.set();
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	found->second.clear_mask.set(slice, is_clear);
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	bool             registered = false;
	ForEachImageInRegion(address, size, [&](ImageId, Image&) {
		registered = true;
		return true;
	});
	return registered;
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	m_surface_metas.erase(m_surface_metas.lower_bound(address),
	                      m_surface_metas.lower_bound(address + size));
	Common::SmallVector<ImageId, 16> deleted_images;
	ForEachImageInRegion(address, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
	for (const auto id: deleted_images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool     pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool     aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		// without memory pressure, keep images for ~5 s at 60 FPS instead of 16
		// frames; at 17 FPS 16 frames is under a second. KYTY_GC_OLD_LIMITS=1 restores 16.
		static const bool old_limits = std::getenv("KYTY_GC_OLD_LIMITS") != nullptr;
		uint64_t age        = old_limits ? 16 : 300;
		size_t   deletions  = 10;
		if (aggressive) {
			age       = 160;
			deletions = 40;
		} else if (pressured) {
			age       = 80;
			deletions = 20;
		}
		age = std::min(age, tick);
		std::vector<ImageId> candidates;
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) { candidates.push_back(id); });
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				continue;
			}
			if (owner->heap_refs.load(std::memory_order_relaxed) != 0u) {
				// Resident heaps sample it without touching it.
				TouchImage(*owner);
				continue;
			}
			if (owner->SafeToDownload() &&
			    (!pressured || owner->info.HasStencil() || !DownloadImageMemory(id))) {
				continue;
			}
			--deletions;
			FreeImage(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureEvictions);
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
}

void TextureCache::ProcessDownloadImages() {
	if (!m_readback_linear_images && !m_readback_tiled_images) {
		return;
	}
	(void)DownloadQueuedImages();
}

bool TextureCache::WriteBackImages() {
	return m_readback_storage_images && DownloadQueuedImages();
}

bool TextureCache::DownloadQueuedImages() {
	std::scoped_lock lock {m_lock};
	bool             downloaded = false;
	for (const auto id: m_download_images) {
		if (m_slot_images[id].IsGpuModified()) {
			downloaded |= DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
	return downloaded;
}

} // namespace Libs::Graphics
