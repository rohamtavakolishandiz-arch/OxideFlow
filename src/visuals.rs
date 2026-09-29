use crate::backend::WgpuBackend;
use crate::renderer::GpuRenderer;
use crate::flappy::FlappyEnv;

pub fn render_frame(
    device: &WgpuBackend,
    surface: &wgpu::Surface,
    config: &wgpu::SurfaceConfiguration,
    renderer: &GpuRenderer,
    env: &FlappyEnv,
    state: &[f64],
    q_data: &[f64],
    action: usize,
    epsilon: f64,
    memory_len: usize,
    batch_size: usize,
    is_low_quality: bool,
) -> bool {
    let mut vertices = Vec::new();
    let w = config.width as f32;
    let h = config.height as f32;
    let game_w = w * 0.58;
    let hud_w = w * 0.42;
    let hud_x = game_w;

    // ========================================================
    // 1. BACKGROUND & SIMULATION GRID
    // ========================================================
    // Dark game canvas and tech HUD partition
    GpuRenderer::push_rect(&mut vertices, 0.0, 0.0, game_w, h, [0.03, 0.04, 0.07, 1.0], w, h);
    GpuRenderer::push_rect(&mut vertices, hud_x, 0.0, hud_w, h, [0.015, 0.02, 0.035, 1.0], w, h);
    GpuRenderer::push_line(&mut vertices, hud_x, 0.0, hud_x, h, 2.0, [0.12, 0.35, 0.65, 0.8], w, h);

    if !is_low_quality {
        // Subtle cyber matrix grid across game viewport
        let grid_step = 40.0;
        let mut gx = 0.0;
        while gx < game_w {
            GpuRenderer::push_line(&mut vertices, gx, 0.0, gx, h, 1.0, [0.06, 0.09, 0.15, 0.4], w, h);
            gx += grid_step;
        }
        let mut gy = 0.0;
        while gy < h {
            GpuRenderer::push_line(&mut vertices, 0.0, gy, game_w, gy, 1.0, [0.06, 0.09, 0.15, 0.4], w, h);
            gy += grid_step;
        }
    }

    // ========================================================
    // 2. OBSTACLES (NEON EDGE-LIT PIPES)
    // ========================================================
    let bird_px_x = 0.2 * game_w;
    let bird_px_y = (env.bird_y * h).clamp(10.0, h - 10.0);
    let mut closest_pipe = None;

    for pipe in &env.pipes {
        let px = pipe.x * game_w;
        let pw = 0.09 * game_w;
        let top_h = pipe.top_gap * h;
        let bot_y = pipe.bottom_gap * h;
        let bot_h = h - bot_y;

        // Pipe Body (Dark Metal)
        GpuRenderer::push_rect(&mut vertices, px, 0.0, pw, top_h, [0.08, 0.14, 0.18, 0.95], w, h);
        GpuRenderer::push_rect(&mut vertices, px, bot_y, pw, bot_h, [0.08, 0.14, 0.18, 0.95], w, h);

        // Neon Trim Edges
        let neon_color = [0.0, 0.88, 0.55, 0.9];
        GpuRenderer::push_line(&mut vertices, px, 0.0, px, top_h, 1.5, neon_color, w, h);
        GpuRenderer::push_line(&mut vertices, px + pw, 0.0, px + pw, top_h, 1.5, neon_color, w, h);
        GpuRenderer::push_line(&mut vertices, px, bot_y, px, h, 1.5, neon_color, w, h);
        GpuRenderer::push_line(&mut vertices, px + pw, bot_y, px + pw, h, 1.5, neon_color, w, h);

        // Pipe Collars / Lips
        let lip_w = pw + 8.0;
        let lip_x = px - 4.0;
        GpuRenderer::push_rect(&mut vertices, lip_x, top_h - 12.0, lip_w, 12.0, [0.0, 0.75, 0.45, 1.0], w, h);
        GpuRenderer::push_rect(&mut vertices, lip_x, bot_y, lip_w, 12.0, [0.0, 0.75, 0.45, 1.0], w, h);

        // Identify the upcoming active target pipe for LiDAR sightlines
        if (px + pw) >= (bird_px_x - 10.0) && closest_pipe.is_none() {
            closest_pipe = Some((px, pw, top_h, bot_y));
        }
    }

    // ========================================================
    // 3. AI SENSOR RAYCASTS (VISION LASERS)
    // ========================================================
    if !is_low_quality {
        if let Some((px, pw, top_h, bot_y)) = closest_pipe {
            let target_gap_center = (top_h + bot_y) * 0.5;

            // Gap center trajectory laser (Cyan)
            GpuRenderer::push_line(&mut vertices, bird_px_x, bird_px_y, px + (pw * 0.5), target_gap_center, 1.5, [0.0, 0.8, 1.0, 0.5], w, h);
            GpuRenderer::push_node(&mut vertices, px + (pw * 0.5), target_gap_center, 4.0, [0.0, 1.0, 0.9, 0.8], w, h);

            // Ceiling hazard sensor (Red)
            GpuRenderer::push_line(&mut vertices, bird_px_x, bird_px_y, px, top_h, 1.0, [1.0, 0.25, 0.35, 0.35], w, h);

            // Floor hazard sensor (Red)
            GpuRenderer::push_line(&mut vertices, bird_px_x, bird_px_y, px, bot_y, 1.0, [1.0, 0.25, 0.35, 0.35], w, h);
        }
    }

    // ========================================================
    // 4. AGENT (BIRD VISUAL WITH BLOOM AURA)
    // ========================================================
    // Multi-pass bloom halo
    GpuRenderer::push_node(&mut vertices, bird_px_x, bird_px_y, 22.0, [1.0, 0.8, 0.1, 0.15], w, h);
    GpuRenderer::push_node(&mut vertices, bird_px_x, bird_px_y, 16.0, [1.0, 0.85, 0.2, 0.35], w, h);
    // Core body
    let agent_core_color = if action == 1 { [1.0, 0.35, 0.15, 1.0] } else { [1.0, 0.88, 0.2, 1.0] };
    GpuRenderer::push_node(&mut vertices, bird_px_x, bird_px_y, 11.0, agent_core_color, w, h);
    // Cyber Visor Eye (Directional point)
    GpuRenderer::push_node(&mut vertices, bird_px_x + 5.0, bird_px_y - 2.0, 3.5, [0.1, 0.95, 1.0, 1.0], w, h);

    // ========================================================
    // 5. TELEMETRY STATUS BARS
    // ========================================================
    let bar_x = hud_x + 25.0;
    let bar_w = hud_w - 50.0;

    // Exploration (Epsilon) Bar
    GpuRenderer::push_rect(&mut vertices, bar_x, 28.0, bar_w, 8.0, [0.08, 0.1, 0.16, 1.0], w, h);
    GpuRenderer::push_rect(&mut vertices, bar_x, 28.0, bar_w * (epsilon as f32), 8.0, [0.75, 0.2, 0.95, 0.9], w, h);

    // Replay Buffer Capacity Bar
    let buffer_pct = (memory_len as f32 / batch_size.max(1) as f32).min(1.0);
    GpuRenderer::push_rect(&mut vertices, bar_x, 44.0, bar_w, 8.0, [0.08, 0.1, 0.16, 1.0], w, h);
    GpuRenderer::push_rect(&mut vertices, bar_x, 44.0, bar_w * buffer_pct, 8.0, [0.0, 0.85, 0.75, 0.9], w, h);

    // ========================================================
    // 6. DEEP NEURAL NETWORK VISUALIZER (INPUT -> HIDDEN -> OUTPUT)
    // ========================================================
    if !is_low_quality {
        let in_x = hud_x + (hud_w * 0.18);
        let hid_x = hud_x + (hud_w * 0.52);
        let out_x = hud_x + (hud_w * 0.84);

        let input_count = 5;
        let hidden_count = 7;
        let output_count = 2;

        let compute_y = |idx: usize, total: usize, start_y: f32, spacing: f32| -> f32 {
            let offset = (total as f32 - 1.0) * spacing * 0.5;
            start_y - offset + (idx as f32 * spacing)
        };

        let center_y = h * 0.44;

        // --- Synapses: Layer 1 (Input to Latent Hidden) ---
        for i in 0..input_count {
            let y1 = compute_y(i, input_count, center_y, h * 0.08);
            let in_val = if i < state.len() { state[i].abs().min(1.0) as f32 } else { 0.2 };

            for j in 0..hidden_count {
                let y2 = compute_y(j, hidden_count, center_y, h * 0.055);
                let alpha = (0.05 + 0.25 * in_val).clamp(0.04, 0.3);
                GpuRenderer::push_line(&mut vertices, in_x, y1, hid_x, y2, 1.2, [0.18, 0.45, 0.85, alpha], w, h);
            }
        }

        // --- Synapses: Layer 2 (Latent Hidden to Output) ---
        for j in 0..hidden_count {
            let y1 = compute_y(j, hidden_count, center_y, h * 0.055);
            for k in 0..output_count {
                let y2 = compute_y(k, output_count, center_y, h * 0.16);
                let is_chosen = action == k;
                let color = if is_chosen {
                    [1.0, 0.55, 0.15, 0.45]
                } else {
                    [0.2, 0.4, 0.6, 0.12]
                };
                GpuRenderer::push_line(&mut vertices, hid_x, y1, out_x, y2, if is_chosen { 2.0 } else { 1.2 }, color, w, h);
            }
        }

        // --- Input Neurons ---
        for i in 0..input_count {
            let y = compute_y(i, input_count, center_y, h * 0.08);
            let val = if i < state.len() { state[i].abs().min(1.0) as f32 } else { 0.5 };
            // Outer halo
            GpuRenderer::push_node(&mut vertices, in_x, y, 12.0, [0.0, 0.5, 1.0, 0.2 * val], w, h);
            // Core
            GpuRenderer::push_node(&mut vertices, in_x, y, 6.5, [0.1, 0.75 + (0.25 * val), 1.0, 1.0], w, h);
        }

        // --- Hidden Layer Neurons ---
        for j in 0..hidden_count {
            let y = compute_y(j, hidden_count, center_y, h * 0.055);
            let pulse = ((j as f32 * 0.7).sin() * 0.5 + 0.5).clamp(0.2, 0.9);
            GpuRenderer::push_node(&mut vertices, hid_x, y, 9.0, [0.4, 0.2, 0.8, 0.25 * pulse], w, h);
            GpuRenderer::push_node(&mut vertices, hid_x, y, 5.0, [0.65, 0.35, 1.0, 0.85], w, h);
        }

        // --- Output Neurons (Q-Values) ---
        let max_q = q_data[0].max(q_data[1]);
        let exp0 = (q_data[0] - max_q).exp() as f32;
        let exp1 = (q_data[1] - max_q).exp() as f32;
        let sum_exp = (exp0 + exp1).max(0.001);
        let prob = [exp0 / sum_exp, exp1 / sum_exp];

        for k in 0..output_count {
            let y = compute_y(k, output_count, center_y, h * 0.16);
            let is_active = action == k;

            if is_active {
                // Expanding target halo
                GpuRenderer::push_node(&mut vertices, out_x, y, 22.0, [1.0, 0.45, 0.1, 0.2], w, h);
                GpuRenderer::push_node(&mut vertices, out_x, y, 16.0, [1.0, 0.5, 0.15, 0.45], w, h);
                GpuRenderer::push_node(&mut vertices, out_x, y, 9.5, [1.0, 0.8, 0.3, 1.0], w, h);
            } else {
                GpuRenderer::push_node(&mut vertices, out_x, y, 7.5, [0.22, 0.26, 0.36, 0.8], w, h);
            }

            // ========================================================
            // 7. REAL-TIME CONFIDENCE METERS
            // ========================================================
            let meter_y = h * 0.76 + (k as f32 * 32.0);
            let meter_w = hud_w - 60.0;
            let meter_x = hud_x + 30.0;

            // Background tray
            GpuRenderer::push_rect(&mut vertices, meter_x, meter_y, meter_w, 14.0, [0.06, 0.08, 0.12, 0.9], w, h);

            // Fill Bar
            let fill_w = meter_w * prob[k];
            let bar_color = if k == 1 {
                if is_active { [1.0, 0.35, 0.15, 0.95] } else { [0.55, 0.2, 0.1, 0.6] }
            } else {
                if is_active { [0.05, 0.8, 0.95, 0.95] } else { [0.08, 0.4, 0.55, 0.6] }
            };
            GpuRenderer::push_rect(&mut vertices, meter_x, meter_y, fill_w, 14.0, bar_color, w, h);

            // Active action pip indicator
            if is_active {
                GpuRenderer::push_rect(&mut vertices, meter_x - 8.0, meter_y, 4.0, 14.0, [1.0, 0.9, 0.2, 1.0], w, h);
            }
        }
    }

    // ========================================================
    // 8. PRESENT PASS
    // ========================================================
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(output) | wgpu::CurrentSurfaceTexture::Suboptimal(output) => {
            let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
            renderer.render(&device.device, &device.queue, &view, &vertices);
            device.queue.present(output);
            false
        }
        _ => true,
    }
}