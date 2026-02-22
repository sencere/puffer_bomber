#pragma once

static void tb_render_impl(TileBlast* env) {
    if (!env) return;

    const int tile = 32;
    const int ww = env->width * tile;
    const int wh = env->height * tile;
    const int pulse = env->tick % 24;
    const int pulse_up = pulse < 12 ? pulse : (24 - pulse);

    if (!IsWindowReady()) {
        InitWindow(ww, wh + 44, "TileBlast");
        SetTargetFPS(18);
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){8, 10, 16, 255});
    DrawRectangleGradientV(0, 0, ww, wh, (Color){20, 24, 38, 255}, (Color){8, 10, 16, 255});

    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int x = c * tile;
            int y = r * tile;

            unsigned char t = env->grid[tb_idx(env, r, c)];
            Color base = ((r + c) & 1) ? (Color){18, 24, 38, 255} : (Color){22, 29, 44, 255};
            DrawRectangle(x, y, tile, tile, base);
            DrawRectangleLines(x, y, tile, tile, (Color){35, 43, 64, 180});

            if (t == TILE_HARD) {
                DrawRectangle(x + 3, y + 3, tile - 6, tile - 6, (Color){84, 96, 120, 255});
                DrawRectangle(x + 6, y + 6, tile - 12, tile - 12, (Color){58, 68, 90, 255});
            } else if (t == TILE_SOFT) {
                DrawRectangle(x + 3, y + 3, tile - 6, tile - 6, (Color){116, 92, 182, 245});
                DrawRectangle(x + 8, y + 8, tile - 16, tile - 16, (Color){155, 124, 228, 235});
            }

            if (env->blast_timer[tb_idx(env, r, c)] > 0) {
                int inset = 4 + (BLAST_TIME - env->blast_timer[tb_idx(env, r, c)]);
                if (inset > tile / 2 - 2) inset = tile / 2 - 2;
                DrawRectangle(x + inset, y + inset, tile - inset * 2, tile - inset * 2,
                              (Color){255, 176, 72, 220});
                DrawRectangle(x + tile / 2 - 2, y + 4, 4, tile - 8, (Color){255, 234, 170, 220});
                DrawRectangle(x + 4, y + tile / 2 - 2, tile - 8, 4, (Color){255, 234, 170, 220});
            }
        }
    }

    {
        int gx = env->goal_col * tile;
        int gy = env->goal_row * tile;
        int glow = 110 + pulse_up * 9;
        if (glow > 255) glow = 255;
        DrawRectangle(gx + 2, gy + 2, tile - 4, tile - 4, (Color){45, 160, 90, 120});
        DrawRectangle(gx + 7, gy + 7, tile - 14, tile - 14, (Color){78, 230, 130, 230});
        DrawCircle(gx + tile / 2, gy + tile / 2, 4 + pulse_up / 4, (Color){160, 255, 190, (unsigned char)glow});
    }

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        int cx = b->col * tile + tile / 2;
        int cy = b->row * tile + tile / 2;
        int fuse_glow = 160 + (pulse_up * 7);
        if (fuse_glow > 255) fuse_glow = 255;
        DrawCircle(cx, cy + 3, 10, (Color){26, 28, 38, 255});
        DrawCircle(cx, cy + 2, 8, (Color){40, 44, 58, 255});
        DrawCircle(cx + 7, cy - 8, 3, (Color){255, 208, 76, (unsigned char)fuse_glow});
        DrawCircleLines(cx, cy + 2, 9, (Color){215, 226, 245, 130});
    }

    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (!e->alive) continue;
        int cx = e->col * tile + tile / 2;
        int cy = e->row * tile + tile / 2;
        int aura = 80 + pulse_up * 6;
        if (aura > 200) aura = 200;
        DrawCircle(cx, cy + 12, 8, (Color){20, 14, 22, 80});
        DrawCircle(cx, cy, 12, (Color){255, 130, 146, (unsigned char)aura});
        DrawCircle(cx, cy + 1, 10, (Color){236, 98, 106, 255});
        DrawTriangle(
            (Vector2){(float)(cx - 8), (float)(cy - 6)},
            (Vector2){(float)(cx - 3), (float)(cy - 15)},
            (Vector2){(float)(cx - 1), (float)(cy - 6)},
            (Color){206, 74, 88, 255});
        DrawTriangle(
            (Vector2){(float)(cx + 8), (float)(cy - 6)},
            (Vector2){(float)(cx + 3), (float)(cy - 15)},
            (Vector2){(float)(cx + 1), (float)(cy - 6)},
            (Color){206, 74, 88, 255});
        DrawCircle(cx - 3, cy - 2, 2, WHITE);
        DrawCircle(cx + 3, cy - 2, 2, WHITE);
        DrawCircle(cx - 3, cy - 2, 1, (Color){28, 20, 20, 255});
        DrawCircle(cx + 3, cy - 2, 1, (Color){28, 20, 20, 255});
        DrawRectangle(cx - 3, cy + 3, 6, 2, (Color){88, 24, 32, 210});
    }

    if (env->agents[0].alive) {
        int cx = env->agents[0].col * tile + tile / 2;
        int cy = env->agents[0].row * tile + tile / 2;
        DrawCircle(cx, cy + 12, 8, (Color){18, 14, 10, 90});
        DrawCircle(cx, cy + 1, 12, (Color){255, 231, 156, 255});
        DrawCircle(cx, cy + 1, 10, (Color){248, 214, 118, 255});
        DrawCircle(cx - 3, cy - 2, 2, (Color){70, 56, 32, 255});
        DrawCircle(cx + 3, cy - 2, 2, (Color){70, 56, 32, 255});
        DrawCircle(cx, cy + 3, 3, (Color){255, 240, 188, 130});
        DrawCircleLines(cx, cy + 1, 12, (Color){255, 249, 220, 160});
        DrawTriangle(
            (Vector2){(float)(cx - 4), (float)(cy + 8)},
            (Vector2){(float)(cx + 4), (float)(cy + 8)},
            (Vector2){(float)cx, (float)(cy + 13)},
            (Color){224, 173, 86, 255});
    }

    DrawRectangle(0, wh, ww, 44, (Color){6, 8, 14, 255});
    DrawLine(0, wh, ww, wh, (Color){44, 52, 76, 255});
    DrawText(TextFormat("Step %d/%d", env->tick, env->max_steps), 10, wh + 12, 18, (Color){228, 232, 244, 255});
    DrawText("Arrows/WASD move, SPACE bomb, ESC quit", 170, wh + 12, 18, (Color){170, 182, 212, 255});

    if (env->outcome_banner_ticks > 0) {
        const char* label = NULL;
        Color color = WHITE;
        if (env->last_outcome == TB_OUTCOME_WIN) {
            label = "WIN";
            color = (Color){90, 235, 140, 255};
        } else if (env->last_outcome == TB_OUTCOME_TIMEOUT) {
            label = "TIMEOUT";
            color = (Color){255, 210, 120, 255};
        } else if (env->last_outcome == TB_OUTCOME_DEAD) {
            label = "DEAD";
            color = (Color){255, 120, 130, 255};
        }

        if (label) {
            int font_size = 34;
            int tw = MeasureText(label, font_size);
            int tx = (ww - tw) / 2;
            int ty = 8;
            DrawRectangle(tx - 14, ty - 8, tw + 28, font_size + 16, (Color){12, 14, 22, 180});
            DrawRectangleLines(tx - 14, ty - 8, tw + 28, font_size + 16, (Color){220, 228, 245, 110});
            DrawText(label, tx, ty, font_size, color);
        }
    }

    EndDrawing();
}
