"""Gradio interface. Preview is a separate step from the full export."""

from __future__ import annotations

import gradio as gr

from mapconv.device import device_summary, recommend_encoder
from mapconv.media import probe
from mapconv.pipeline import (
    format_error,
    generate_images,
    generate_video,
    preview_image,
    preview_video,
)
from mapconv.recipe import (
    BLEND_CHOICES,
    ENCODER_CHOICES,
    MAP_LABELS,
    Recipe,
    maps_to_layers,
)

MAX_SIDE_CHOICES = [
    ("720", 720),
    ("960", 960),
    ("1280", 1280),
    ("1920", 1920),
    ("Original", 0),
]
SOFT_CHOICES = [("HED (thicker)", "hed"), ("PiDiNet (finer)", "pidinet")]


def _path(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return _path(value[0]) if value else None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    name = getattr(value, "name", None)
    return str(name) if name else None


def _paths(value) -> list[str]:
    if value is None or value == "":
        return []
    if not isinstance(value, (list, tuple)):
        found = _path(value)
        return [found] if found else []
    found = []
    for item in value:
        path = _path(item)
        if path:
            found.append(path)
    return found


def _recipe(
    layer1,
    layer1_opacity,
    layer1_mode,
    layer2,
    layer2_opacity,
    layer2_mode,
    layer3,
    layer3_opacity,
    layer3_mode,
    layer4,
    layer4_opacity,
    layer4_mode,
    clay_relief,
    encoder,
    input_size,
    max_side,
    contrast,
    blur,
    invert,
    clip_extremes,
    dither,
    keep_audio,
    crf,
    save_frames,
    save_npz,
    max_seconds,
    target_fps,
    canny_low,
    canny_high,
    lineart_coarse,
    mlsd_score,
    mlsd_distance,
    softedge,
    pose_hands,
    pose_face,
    detect_resolution,
) -> Recipe:
    return Recipe(
        layer1=layer1 or "depth",
        layer1_opacity=float(layer1_opacity),
        layer1_mode=layer1_mode or "blend",
        layer2=layer2 or "none",
        layer2_opacity=float(layer2_opacity),
        layer2_mode=layer2_mode or "transparent",
        layer3=layer3 or "none",
        layer3_opacity=float(layer3_opacity),
        layer3_mode=layer3_mode or "transparent",
        layer4=layer4 or "none",
        layer4_opacity=float(layer4_opacity),
        layer4_mode=layer4_mode or "transparent",
        clay_relief=float(clay_relief),
        encoder=encoder or "auto",
        input_size=int(input_size),
        max_side=int(max_side or 0),
        contrast=float(contrast),
        blur=float(blur),
        invert=bool(invert),
        clip_extremes=bool(clip_extremes),
        dither=bool(dither),
        keep_audio=bool(keep_audio),
        crf=int(crf),
        save_frames=bool(save_frames),
        save_npz=bool(save_npz),
        max_seconds=float(max_seconds),
        target_fps=float(target_fps),
        canny_low=int(canny_low),
        canny_high=int(canny_high),
        lineart_coarse=bool(lineart_coarse),
        mlsd_score=float(mlsd_score),
        mlsd_distance=float(mlsd_distance),
        softedge=softedge or "hed",
        pose_hands=bool(pose_hands),
        pose_face=bool(pose_face),
        detect_resolution=int(detect_resolution),
    )


def _settings_visibility(layer1, layer2, layer3, layer4):
    names = {layer1 or "", layer2 or "", layer3 or "", layer4 or ""}
    names.discard("none")
    learned = {"lineart", "lineart_anime", "mlsd", "softedge", "scribble", "pose", "normal"}
    return (
        gr.update(visible="depth" in names or "clay" in names),
        gr.update(visible="clay" in names),
        gr.update(visible="canny" in names),
        gr.update(visible="lineart" in names or "lineart_anime" in names),
        gr.update(visible="mlsd" in names),
        gr.update(visible="softedge" in names or "scribble" in names),
        gr.update(visible="pose" in names),
        gr.update(visible=bool(names & learned)),
    )


def _describe_video(video):
    path = _path(video)
    if not path:
        return "No video loaded.", gr.update()
    try:
        info = probe(path)
    except Exception as exc:
        return format_error(exc), gr.update()
    maximum = max(info.duration - 0.05, 0) if info.duration else 0
    return info.summary(), gr.update(maximum=maximum, value=0)


def _describe_images(images):
    paths = _paths(images)
    if not paths:
        return "No images loaded."
    return f"{len(paths)} image(s). Preview uses the first one. Generate converts all of them."


CHECKBOX_ORDER = [
    "clay",
    "depth",
    "canny",
    "pose",
    "lineart",
    "lineart_anime",
    "mlsd",
    "softedge",
    "scribble",
    "normal",
]


def _checked_maps(*flags: bool) -> list[str]:
    return [name for name, flag in zip(CHECKBOX_ORDER, flags) if flag]


def _on_maps(*values):
    flags = values[:10]
    keys = values[10:14]
    opacities = values[14:18]
    modes = values[18:22]
    selected = _checked_maps(*flags)
    defaults, overflow = maps_to_layers(selected)
    previous = {key: (opacity, mode) for key, opacity, mode in zip(keys, opacities, modes)}
    updates = []
    active = []
    for name, opacity, mode in defaults:
        if name != "none" and name in previous:
            opacity, mode = previous[name]
        shown = name != "none"
        if shown:
            active.append(name)
        updates.append(gr.update(visible=shown))
        updates.append(gr.update(value=MAP_LABELS.get(name, "")))
        updates.append(gr.update(value=name))
        updates.append(gr.update(value=float(opacity)))
        updates.append(gr.update(value=mode))
    if not active:
        note = "Turn on at least one map."
    elif overflow:
        skipped = ", ".join(MAP_LABELS[name] for name in overflow)
        note = f"Four maps are the maximum. Left off: {skipped}."
    else:
        order = ", then ".join(MAP_LABELS[name] for name in active)
        note = f"Stack, bottom to top: {order}."
    updates.append(note)
    updates.extend(_settings_visibility(*[item[0] for item in defaults]))
    return tuple(updates)


def _fail_preview(message):
    return None, None, None, message


def _fail_generate(message):
    return None, None, message


def do_preview(
    source,
    video,
    images,
    start_sec,
    *settings,
    progress=gr.Progress(),
):
    recipe = _recipe(*settings)
    try:
        if source == "Images":
            paths = _paths(images)
            if not paths:
                return _fail_preview("Add one or more images.")
            image, layers, status = preview_image(paths[0], recipe, progress)
            return image, None, layers, status
        path = _path(video)
        if not path:
            return _fail_preview("Add a video.")
        image, video_path, layers, status = preview_video(path, recipe, float(start_sec or 0), progress)
        return image, video_path, layers, status
    except Exception as exc:
        return _fail_preview(format_error(exc))


def do_generate(source, video, images, *settings, progress=gr.Progress()):
    recipe = _recipe(*settings)
    try:
        if source == "Images":
            paths = _paths(images)
            if not paths:
                return _fail_generate("Add one or more images.")
            gallery, status = generate_images(paths, recipe, progress)
            return None, gallery, status
        path = _path(video)
        if not path:
            return _fail_generate("Add a video.")
        video_path, status = generate_video(path, recipe, progress)
        return video_path, None, status
    except Exception as exc:
        return _fail_generate(format_error(exc))


def build():
    _encoder, suggested_size = recommend_encoder()
    with gr.Blocks(title="Map Converter", theme=gr.themes.Soft(primary_hue="slate")) as demo:
        gr.Markdown(
            f"""
# Map Converter
Turn a video or image into a depth map, and stack up to four maps.
Videos use **Video Depth Anything** so the gray stays on one scale. Images use **Depth Anything V2**.
Near is white unless Invert is on. Preview a short piece before the full export.

**Device:** {device_summary()}
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(["Video", "Images"], value="Video", label="Source")
                video = gr.File(
                    label="Video",
                    file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"],
                    type="filepath",
                )
                images = gr.File(
                    label="Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    visible=False,
                )
                source_info = gr.Markdown("Load a video or images.")
                start_sec = gr.Slider(
                    label="Preview start (seconds)",
                    minimum=0,
                    maximum=60,
                    value=0,
                    step=0.1,
                    info="Preview converts 32 frames from this time with the same model as the full export.",
                )
                gr.Markdown(
                    "Turn maps on or off. They stack **Clay, then Depth, then lines, then Pose**. "
                    "Opacity is how much a layer covers what is under it. "
                    "**Alpha blend** mixes the whole frame. **Transparent overlay** drops the background."
                )
                with gr.Row():
                    use_clay = gr.Checkbox(value=False, label="Clay")
                    use_depth = gr.Checkbox(value=True, label="Depth")
                    use_canny = gr.Checkbox(value=False, label="Canny")
                    use_pose = gr.Checkbox(value=False, label="Pose")
                with gr.Row():
                    use_lineart = gr.Checkbox(value=False, label="Line art")
                    use_lineart_anime = gr.Checkbox(value=False, label="Anime lines")
                    use_mlsd = gr.Checkbox(value=False, label="MLSD")
                    use_softedge = gr.Checkbox(value=False, label="Soft edge")
                    use_scribble = gr.Checkbox(value=False, label="Scribble")
                    use_normal = gr.Checkbox(value=False, label="Normal")
                stack_note = gr.Markdown("Stack, bottom to top: Depth.")
                layer_rows = []
                layer_labels = []
                layer_keys = []
                layer_opacities = []
                layer_modes = []
                initial = [("depth", "Depth", 1.0, "blend", True)] + [("none", "", 0.4, "transparent", False)] * 3
                for index, (key, label, opacity, mode, shown) in enumerate(initial, start=1):
                    with gr.Row(visible=shown) as row:
                        layer_labels.append(gr.Textbox(value=label, label=f"Layer {index}", interactive=False))
                        layer_opacities.append(gr.Slider(0, 1, value=opacity, step=0.05, label="Opacity"))
                        layer_modes.append(gr.Dropdown(BLEND_CHOICES, value=mode, label="Composite"))
                    layer_rows.append(row)
                    layer_keys.append(gr.Textbox(value=key, visible=False))
                layer1, layer2, layer3, layer4 = layer_keys
                layer1_opacity, layer2_opacity, layer3_opacity, layer4_opacity = layer_opacities
                layer1_mode, layer2_mode, layer3_mode, layer4_mode = layer_modes

                with gr.Accordion("Depth", open=True) as depth_box:
                    encoder = gr.Dropdown(
                        ENCODER_CHOICES,
                        value="auto",
                        label="Depth model",
                        info="Small is Apache-2.0. Base and Large weights are CC-BY-NC-4.0 (non-commercial). Auto picks a size that fits this GPU.",
                    )
                    input_size = gr.Slider(
                        280,
                        700,
                        value=suggested_size,
                        step=14,
                        label="Depth input size",
                        info="Lower this if the GPU runs out of memory. 518 is the upstream default. The output stays at Max resolution.",
                    )
                    clip_extremes = gr.Checkbox(
                        value=True,
                        label="Ignore extreme depths",
                        info="Fit gray to the 1st-99th percentile.",
                    )

                with gr.Accordion("Clay", open=False, visible=False) as clay_box:
                    clay_relief = gr.Slider(
                        0,
                        120,
                        value=40,
                        step=1,
                        label="Clay relief",
                        info="How strongly the depth is shaded. Higher looks more modeled. Clay is lit from the upper left.",
                    )

                with gr.Accordion("Look", open=True):
                    max_side = gr.Dropdown(MAX_SIDE_CHOICES, value=1280, label="Max resolution")
                    contrast = gr.Slider(
                        0.5,
                        2.5,
                        value=1.0,
                        step=0.05,
                        label="Depth contrast",
                        info="Higher locks the depth silhouettes. Lower lets a generator invent more. Clay uses Clay relief instead.",
                    )
                    blur = gr.Slider(
                        0,
                        8,
                        value=0,
                        step=0.1,
                        label="Depth blur",
                        info="Softens the depth layer. Line and pose layers stay sharp.",
                    )
                    invert = gr.Checkbox(value=False, label="Invert (near = black)")
                    dither = gr.Checkbox(
                        value=True,
                        label="Dither 8-bit output",
                        info="Reduces banding when the gray map is saved as H.264.",
                    )

                with gr.Accordion("Canny", open=False, visible=False) as canny_box:
                    canny_low = gr.Slider(1, 400, value=100, step=1, label="Canny low threshold")
                    canny_high = gr.Slider(1, 500, value=200, step=1, label="Canny high threshold")
                    gr.Markdown("Raise both thresholds until texture and text drop out and the silhouette remains.")

                with gr.Accordion("Line art", open=False, visible=False) as lineart_box:
                    lineart_coarse = gr.Checkbox(
                        value=True,
                        label="Coarse line art",
                        info="Coarse drops interior detail. Turn it off for a finer drawing.",
                    )

                with gr.Accordion("MLSD", open=False, visible=False) as mlsd_box:
                    mlsd_score = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Score threshold")
                    mlsd_distance = gr.Slider(0.01, 20, value=0.1, step=0.01, label="Distance threshold")

                with gr.Accordion("Soft edge / scribble", open=False, visible=False) as soft_box:
                    softedge = gr.Dropdown(SOFT_CHOICES, value="hed", label="Soft edge model")

                with gr.Accordion("Pose", open=False, visible=False) as pose_box:
                    pose_hands = gr.Checkbox(value=False, label="Include hands")
                    pose_face = gr.Checkbox(value=False, label="Include face")
                    gr.Markdown("Body only is the cleaner overlay. Add hands and face only when those need to track.")

                with gr.Accordion("Other maps", open=False, visible=False) as aux_box:
                    detect_resolution = gr.Slider(
                        256,
                        1280,
                        value=512,
                        step=64,
                        label="Detector resolution",
                        info="Learned line, pose, and normal models run at this size, then the result is scaled to the frame.",
                    )

                with gr.Accordion("Export", open=False):
                    keep_audio = gr.Checkbox(value=True, label="Keep original audio")
                    crf = gr.Slider(
                        8,
                        23,
                        value=14,
                        step=1,
                        label="H.264 CRF",
                        info="Lower is a larger file with less banding. 14 is a high-bitrate default.",
                    )
                    max_seconds = gr.Slider(
                        0,
                        300,
                        value=60,
                        step=1,
                        label="Max seconds",
                        info="0 converts the whole file. The default stops at 60 seconds so a long file cannot fill memory.",
                    )
                    target_fps = gr.Number(
                        value=-1,
                        label="Target FPS",
                        info="-1 keeps the source frame rate.",
                    )
                    save_frames = gr.Checkbox(value=False, label="Also save a PNG sequence")
                    save_npz = gr.Checkbox(value=False, label="Also save raw float depth (.npz)")

                with gr.Row():
                    preview_btn = gr.Button("Preview", variant="secondary")
                    generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                preview_image_out = gr.Image(label="Preview frame", type="numpy")
                preview_video_out = gr.Video(label="Preview clip", autoplay=True, loop=True)
                layer_gallery = gr.Gallery(label="Layers in the preview", columns=4, height=280)
                output_video = gr.Video(label="Export")
                output_gallery = gr.Gallery(label="Exported images", columns=4, height=280)
                status = gr.Textbox(label="Status", lines=12)

        with gr.Accordion("Using the file", open=False):
            gr.Markdown(
                """
Upload the map as the video reference and a character or scene still as the image reference.

Suggested prompt: the video is a blockout. It provides only motion, camera, and spatial structure. Do not use its gray appearance, materials, or scene. The image defines character, materials, color, and lighting.

Check Clay, Depth, and Canny together for that stack: clay on the bottom, depth over it with the far background removed, and Canny lines on top. Add Pose when limbs cross. One continuous shot behaves better than a clip with cuts.
                """
            )

        settings = [
            layer1,
            layer1_opacity,
            layer1_mode,
            layer2,
            layer2_opacity,
            layer2_mode,
            layer3,
            layer3_opacity,
            layer3_mode,
            layer4,
            layer4_opacity,
            layer4_mode,
            clay_relief,
            encoder,
            input_size,
            max_side,
            contrast,
            blur,
            invert,
            clip_extremes,
            dither,
            keep_audio,
            crf,
            save_frames,
            save_npz,
            max_seconds,
            target_fps,
            canny_low,
            canny_high,
            lineart_coarse,
            mlsd_score,
            mlsd_distance,
            softedge,
            pose_hands,
            pose_face,
            detect_resolution,
        ]
        visibility_outputs = [depth_box, clay_box, canny_box, lineart_box, mlsd_box, soft_box, pose_box, aux_box]
        map_checks = [
            use_clay,
            use_depth,
            use_canny,
            use_pose,
            use_lineart,
            use_lineart_anime,
            use_mlsd,
            use_softedge,
            use_scribble,
            use_normal,
        ]
        map_outputs = []
        for row, label, key, opacity, mode in zip(layer_rows, layer_labels, layer_keys, layer_opacities, layer_modes):
            map_outputs.extend([row, label, key, opacity, mode])
        map_outputs.append(stack_note)
        map_outputs.extend(visibility_outputs)

        def _source_change(choice):
            return (
                gr.update(visible=choice == "Video"),
                gr.update(visible=choice == "Images"),
                gr.update(visible=choice == "Video"),
            )

        source.change(_source_change, inputs=source, outputs=[video, images, start_sec])
        video.change(_describe_video, inputs=video, outputs=[source_info, start_sec])
        images.change(_describe_images, inputs=images, outputs=source_info)
        for check in map_checks:
            check.change(
                _on_maps,
                inputs=[*map_checks, *layer_keys, *layer_opacities, *layer_modes],
                outputs=map_outputs,
            )
        preview_btn.click(
            do_preview,
            inputs=[source, video, images, start_sec, *settings],
            outputs=[preview_image_out, preview_video_out, layer_gallery, status],
            concurrency_limit=1,
        )
        generate_btn.click(
            do_generate,
            inputs=[source, video, images, *settings],
            outputs=[output_video, output_gallery, status],
            concurrency_limit=1,
        )
    return demo
