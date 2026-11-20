# KytyPS5 — AMD fixes (RDNA2, Windows amdvlk + Linux RADV)

This branch is a fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5) (base commit `bc934ef0`) with fixes and
diagnostics for running Unreal Engine titles on an AMD Radeon RX 6800. All credit for the emulator itself goes
to the KytyPS5 authors; this fork only adds the changes listed below. Same license as upstream: **GPL-2.0**
(see `LICENSE`).

No game files, keys or firmware are included or linked. You need your own legally obtained content.

## Test setup

- GPU: Radeon RX 6800 (16 GB), 6-core CPU.
- Windows 11, AMD Adrenalin 26.10.1 (proprietary Vulkan driver / amdvlk compiler).
- Ubuntu 26.04, Mesa RADV.
- Titles tested: Dragon Ball Sparking! Zero (PPSA15210), Kena: Bridge of Spirits (PPSA01746),
  Police Simulator (PPSA05831), Cult of the Lamb (PPSA06465), Astro Bot (PPSA21564).

Every fix has an environment switch to turn it off, so each one can be A/B compared.

## Fixes

1. **Mesh shader `gl_Layer` / `ViewportIndex` inside a per-primitive Block.** amdvlk crashes when they are loose
   arrays. Fixed the Police Simulator and Kena crashes.
2. **BC5/BC6H storage.** The driver rejects storage usage for them; the image is created through a format with the
   same block size (BC7). Police went from black-and-white to correct menus.
3. **Tiler:** a chained `?:` on `uvec4` with a specialization constant becomes a vector `OpSpecConstantOp Select`
   that crashes amdvlk (bug #7 in KytyPS5#1624). Using if/return instead fixed Kena's New Game crash.
4. **Tonemap LUT drawn in 8-layer batches.** amdvlk and RADV report `maxMeshOutputLayers = 8`. 32/64-layer LUTs
   were truncated, which caused the "black and white" Unreal image on AMD. Kena renders in color on both drivers.
5. **DCC on mutable images.** Pass `VkImageFormatListCreateInfo` with all same-size color formats; otherwise amdvlk
   keeps DCC on and reinterpreting reads produce tile-sized blocks. Switch: `KYTY_DCC_UNSAFE=1`.
6. **`TryConsumeComputeMetaClear` was too permissive.** Unreal's tile classifier wrote into memory a dead surface had
   registered as DCC metadata, so the dispatch was consumed as a metadata clear and never ran; the indirect passes
   got 0 tiles and the HDR target kept stale bytes (the DBZ "rainbow"). A metadata clear may no longer read images
   or use atomics. Switch: `KYTY_LOOSE_META_CLEAR=1`.
7. **Packed-float images (B10G11R11 / E5B9G9R9) no longer share a backing with other 32-bit formats** in the texture
   cache (UE transient pool aliasing). Switch: `KYTY_SHARE_PACKED_FLOAT=1`.
8. **Fresh float storage/render targets start at zero** instead of uploading stale guest bytes that decode as NaN.
   Switch: `KYTY_UPLOAD_NEW_STORAGE=1`.
9. **Float color exports sanitized** (NaN → 0, ±inf clamped to the half range). Switch: `KYTY_KEEP_NAN_EXPORTS=1`.
10. **`s_cbranch_execz` resolved per lane instead of by a subgroup vote.** The AMD proprietary compiler loses per-lane
    values (LDS atomic offsets) merged after a branch decided by `OpGroupNonUniformAny`. In an isolated repro of UE's
    tile classifier, ~6,200 of 18,360 list entries were written with the vote and 18,360/18,360 per lane on every
    run. This fixed the DBZ checkerboard and ghosting. Only the forward EXECZ skip changes (new `ExecSkip`
    condition kind); loop EXECNZ keeps the vote. Switch: `KYTY_WAVE_EXEC_BRANCHES=1`.
11. **Mip clamp past `MAX_MIP`** when no stored chain matches the view (Astro Bot desert looked washed out).
12. **Pipelines compiled on worker threads.** First-time compiles took 150–225 ms each and froze the frame
    ("No frames for 4 s"); like other emulators' async shaders, they now compile in the background.
13. **Texture cache tuned for 16 GB cards** (the eviction trigger fired at ~3 GB while DBZ keeps ~5.4 GB) and the GPU
    submit thread raised above the guest's worker threads on Windows.

## Diagnostics added

- `KYTY_PERF_STATS=1`: every 60 flips prints FPS, GPU waits, draw-thread time, sync readbacks and cache stats.
- `KYTY_DUMP_DIR` + a `trigger` file: dumps every GPU-written color image of the next frame.
- `KYTY_DUMP_CS=<hash,...>`: saves the buffers of those compute shaders after a dump.
- `KYTY_TRACE=WxH[:fmt,...]`: texture-cache events, draw shaders and inputs, indirect dispatch arguments read back
  from the GPU, and dispatches consumed as clears.
- `KYTY_UE_CVARS="a=b,c=d"`: appends Unreal cvars to `uecommandline.txt` without touching the game.
- `KYTY_RD_AFTER_SEC=N[,N2,...]`: timed RenderDoc captures.
- `KYTY_DISABLE_READLANE_ELIMINATION_HASHES=<hex,...>`: skip the ReadLane pass for specific shaders.

## Status (RX 6800)

| Title | Windows (amdvlk) | Linux (RADV) |
|---|---|---|
| Dragon Ball Sparking! Zero | Playable, correct image; ~16–18 FPS (CPU-bound); device lost after the post-tutorial intro | Playable ~20 FPS; occasional block artifacts and device lost |
| Kena | Plays, color correct | Plays, color correct |
| Police Simulator | Plays | — |
| The Exit 9 | Device lost at the logos | — |

Still open: possible ghost shadows in DBZ fights; RenderDoc / GFXReconstruct cause device lost with Kyty on AMD
Windows.

## Building

Same as upstream KytyPS5 (CMake + Ninja, Vulkan SDK). Nothing extra is required.
