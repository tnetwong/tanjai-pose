extends Node3D
# เรนเดอร์ทุกฉากใน shots/list.json ฉากละหลายมุม แล้วบันทึก PNG ลง ../out

func mat(c: Array, rough := 0.55) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(c[0], c[1], c[2]); m.roughness = rough
	return m

func box(size: Vector3, pos: Vector3, c: Array) -> MeshInstance3D:
	var b := BoxMesh.new(); b.size = size
	var mi := MeshInstance3D.new(); mi.mesh = b; mi.position = pos; mi.material_override = mat(c, 0.8)
	return mi

func _ready():
	var env := WorldEnvironment.new(); var e := Environment.new()
	e.background_mode = Environment.BG_COLOR; e.background_color = Color(1, 1, 1)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR; e.ambient_light_color = Color(0.85, 0.87, 0.92); e.ambient_light_energy = 0.4
	e.tonemap_mode = Environment.TONE_MAPPER_FILMIC; env.environment = e; add_child(env)
	var sun := DirectionalLight3D.new(); sun.light_energy = 0.85; sun.rotation_degrees = Vector3(-55, 35, 0)
	sun.shadow_enabled = true; sun.directional_shadow_max_distance = 8; add_child(sun)
	var fill := DirectionalLight3D.new(); fill.light_energy = 0.25; fill.rotation_degrees = Vector3(-20, -140, 0); add_child(fill)
	var floor := MeshInstance3D.new(); var pm := PlaneMesh.new(); pm.size = Vector2(12, 12); floor.mesh = pm
	floor.material_override = mat([0.92, 0.92, 0.92], 0.9); add_child(floor)
	var names = JSON.parse_string(FileAccess.get_file_as_string("res://shots/list.json"))
	var out_dir := ProjectSettings.globalize_path("res://").path_join("../out")
	for n in names:
		var meta = JSON.parse_string(FileAccess.get_file_as_string("res://shots/%s/meta.json" % n))
		var holder := Node3D.new(); add_child(holder)
		for p in meta.parts:
			var mi := MeshInstance3D.new(); mi.mesh = load("res://shots/%s/%s.obj" % [n, p.name])
			mi.material_override = mat(p.color); holder.add_child(mi)
		if meta.bed:
			var bs = meta.bed_size; var bh = meta.bed_h
			holder.add_child(box(Vector3(bs[0], bs[1], bs[2]), Vector3(0, bh - bs[1] / 2, 0), [0.50, 0.58, 0.68]))
			holder.add_child(box(Vector3(bs[0] * 0.9, bh - bs[1], bs[2] * 0.9), Vector3(0, (bh - bs[1]) / 2, 0), [0.55, 0.5, 0.46]))
		var mn := Vector3(meta.min[0], meta.min[1], meta.min[2]); var mx := Vector3(meta.max[0], meta.max[1], meta.max[2])
		var center := (mn + mx) / 2.0; var diag := (mx - mn).length()
		for c in meta.cams:
			var cam := Camera3D.new(); cam.fov = 32; holder.add_child(cam)
			var d := Vector3(c[1][0], c[1][1], c[1][2]).normalized()
			cam.position = center + d * diag * 1.7; cam.look_at(center, Vector3.UP); cam.current = true
			await get_tree().process_frame; await get_tree().process_frame
			await RenderingServer.frame_post_draw
			get_viewport().get_texture().get_image().save_png(out_dir.path_join("%s-%s.png" % [n, c[0]]))
			cam.queue_free()
		holder.queue_free()
		await get_tree().process_frame
	get_tree().quit()
