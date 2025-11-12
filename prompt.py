def build_camera_prompt(rotate_deg, move_forward, vertical_tilt, wideangle):
    prompt_parts = []

    # Rotation
    if rotate_deg != 0:
        direction = "left" if rotate_deg > 0 else "right"
        if direction == "left":
            prompt_parts.append(f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left.")
        else:
            prompt_parts.append(f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right.")


    # Move forward / close-up
    if move_forward > 5:
        prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif move_forward >= 1:
        prompt_parts.append("将镜头向前移动 Move the camera forward.")

    # Vertical tilt
    if vertical_tilt <= -1:
        prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    elif vertical_tilt >= 1:
        prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

    # Lens option
    if wideangle:
        prompt_parts.append(" 将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

    final_prompt = " ".join(prompt_parts).strip()
    return final_prompt if final_prompt else "no camera movement"