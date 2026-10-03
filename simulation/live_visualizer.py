import pygame
import sys
from pathlib import Path
import numpy as np
from collections import deque
from maine.main import SimulationEngine

class LiveVisualizer:
    def __init__(self, config_params=None):
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("FSAE Powertrain Sim - Live Telemetry")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.bold_font = pygame.font.SysFont("Arial", 22, bold=True)

        # Initialize Simulation Engine
        self.sim = SimulationEngine(config_params)
        self.sim._reset_state()
        
        # History for G-G Diagram trail
        self.gg_history = deque(maxlen=40)

        # Manual control state
        self.manual_mode = True
        self.manual_throttle = 0.0
        self.manual_steering = 0.0

        # Scaling math for track visualization
        padding = 60
        track_w = self.sim.track_max_x - self.sim.track_min_x
        track_h = self.sim.track_max_y - self.sim.track_min_y
        self.scale = min((self.width - 2*padding) / track_w, (self.height - 2*padding) / track_h)
        self.offset_x = padding - self.sim.track_min_x * self.scale
        self.offset_y = padding - self.sim.track_min_y * self.scale

    def world_to_screen(self, x, y):
        screen_x = int(x * self.scale + self.offset_x)
        # Flip Y for screen coordinates (Pygame Y is down)
        screen_y = int(self.height - (y * self.scale + self.offset_y))
        return screen_x, screen_y

    def draw_hud(self, throttle, brake, steering, velocity, soc, temp, p_net, limit_status, ax_g, ay_g, tire_data):
        # Background panel for HUD
        pygame.draw.rect(self.screen, (30, 30, 30), (10, 10, 250, 200))
        pygame.draw.rect(self.screen, (100, 100, 100), (10, 10, 250, 200), 2)

        # Text Stats
        stats = [
            (f"Speed: {velocity*2.237:.1f} mph", (255, 255, 255)),
            (f"Net Power: {p_net/1000:.1f} kW", (255, 200, 0)),
            (f"Battery: {soc:.1f} %", (0, 255, 100)),
            (f"Temp: {temp:.1f} C", (255, 100, 0) if temp < 55 else (255, 0, 0)),
            (f"Status: {limit_status}", (200, 200, 200)),
            (f"Mode: {'MANUAL' if self.manual_mode else 'AUTO'}", (255, 255, 0) if self.manual_mode else (0, 200, 255))
        ]
        for i, (text, color) in enumerate(stats):
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (20, 20 + i * 25))

        # Throttle and Brake Bars
        bar_x, bar_y, bar_w, bar_h = 20, 180, 100, 15
        # Throttle (Green)
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        if throttle > 0:
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, int(throttle * bar_w), bar_h))
        
        # Brake (Red)
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x + 120, bar_y, bar_w, bar_h))
        if throttle < 0:
            pygame.draw.rect(self.screen, (255, 0, 0), (bar_x + 120, bar_y, int(-throttle * bar_w), bar_h))
        
        self.screen.blit(self.font.render("Throttle", True, (200, 200, 200)), (bar_x, bar_y - 20))
        self.screen.blit(self.font.render("Brake", True, (200, 200, 200)), (bar_x + 120, bar_y - 20))

        # Steering Visualizer
        steer_center = (1100, 100)
        pygame.draw.circle(self.screen, (100, 100, 100), steer_center, 50, 2)
        angle_rad = -steering # Exaggerate for visual clarity
        end_x = steer_center[0] + 50 * np.sin(angle_rad)
        end_y = steer_center[1] - 50 * np.cos(angle_rad)
        pygame.draw.line(self.screen, (0, 150, 255), steer_center, (end_x, end_y), 4)
        self.screen.blit(self.font.render("Steering Angle", True, (200, 200, 200)), (1050, 160))

        # Friction Circle (G-G Diagram)
        gg_center = (1100, 300)
        gg_radius = 60
        # Drawing the grid (Circles at 1G and Max Lat G)
        pygame.draw.circle(self.screen, (60, 60, 60), gg_center, gg_radius, 1) # Outer limit
        pygame.draw.circle(self.screen, (40, 40, 40), gg_center, int(gg_radius * 0.5), 1) # 0.5G mark
        pygame.draw.line(self.screen, (60, 60, 60), (gg_center[0]-gg_radius, gg_center[1]), (gg_center[0]+gg_radius, gg_center[1]), 1)
        pygame.draw.line(self.screen, (60, 60, 60), (gg_center[0], gg_center[1]-gg_radius), (gg_center[0], gg_center[1]+gg_radius), 1)
        
        # Scale: gg_radius = max_lat_accel_g
        g_scale = gg_radius / self.sim.p.get('max_lat_accel_g', 2.0)

        # Tire Loading & Friction Ellipses
        # Positioning: Below G-G Diagram
        tire_centers = {"Front": (1050, 550), "Rear": (1150, 550)}
        force_scale = 0.04  # 1 pixel = 25 Newtons

        for label, pos in tire_centers.items():
            data = tire_data[label]
            
            # 1. Draw Friction Ellipse (Potential Grip)
            # Radius X = Fz * mu_long, Radius Y = Fz * mu_lat
            rx = int(data['fz'] * self.sim.p['max_grip_accel_g'] * force_scale)
            ry = int(data['fz'] * self.sim.p['max_lat_accel_g'] * force_scale)
            
            # Background Ellipse
            pygame.draw.ellipse(self.screen, (40, 40, 40), (pos[0]-rx, pos[1]-ry, rx*2, ry*2), 1)
            
            # 2. Draw Current Force Vector (Fx, Fy)
            fx_px = int(data['fx'] * force_scale)
            fy_px = int(data['fy'] * force_scale)
            
            # Color based on grip utilization
            utilization = np.hypot(fx_px/max(1, rx), fy_px/max(1, ry))
            dot_color = (255, 255, 0) if utilization < 0.9 else (255, 0, 0)
            
            # Vector line
            pygame.draw.line(self.screen, (150, 150, 150), pos, (pos[0] + fy_px, pos[1] - fx_px), 2)
            # Force Dot
            pygame.draw.circle(self.screen, dot_color, (pos[0] + fy_px, pos[1] - fx_px), 4)
            
            # 3. Contact Patch Load (Vertical Bar)
            bar_w, bar_max_h = 10, 60
            # Normalize against static load (~1500N for a 300kg car)
            load_h = int((data['fz'] / 3000.0) * bar_max_h) 
            pygame.draw.rect(self.screen, (30, 30, 30), (pos[0] - 60, pos[1] - 30, bar_w, bar_max_h))
            pygame.draw.rect(self.screen, (0, 150, 255), (pos[0] - 60, pos[1] + 30 - load_h, bar_w, load_h))
            
            # Labels
            lbl_surf = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(lbl_surf, (pos[0] - 20, pos[1] + ry + 10))
            
            load_val = self.font.render(f"{int(data['fz'])}N", True, (0, 150, 255))
            self.screen.blit(load_val, (pos[0] - 75, pos[1] + 35))

        self.screen.blit(self.font.render("Friction Ellipses & Load", True, (200, 200, 200)), (1030, 670))

        # Plot Trail
        '''
        for i, (h_ax, h_ay) in enumerate(self.gg_history):
            alpha = int(255 * (i / len(self.gg_history)))
            h_x = gg_center[0] + (h_ay * g_scale)
            h_y = gg_center[1] - (h_ax * g_scale)
            pygame.draw.circle(self.screen, (alpha, alpha, 0), (int(h_x), int(h_y)), 2)'''

        # Plot the G-Dot
        dot_x = gg_center[0] + (ay_g * g_scale)
        dot_y = gg_center[1] - (ax_g * g_scale) # Pygame Y is down, so subtract ax
        pygame.draw.circle(self.screen, (255, 255, 0), (int(dot_x), int(dot_y)), 5)
        self.screen.blit(self.font.render("G-G Diagram", True, (200, 200, 200)), (1060, 370))

    def run(self):
        running = True
        t = 0.0
        steps_per_frame = 2 # Simulation dt is 0.008, so 2 steps ~ 0.016s (approx 60FPS)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.manual_mode = not self.manual_mode
                        print(f"Control Mode: {'MANUAL' if self.manual_mode else 'AUTO'}")

            # Process Keyboard Inputs
            keys = pygame.key.get_pressed()
            if self.manual_mode:
                # Throttle/Brake processing
                if keys[pygame.K_UP]: self.manual_throttle = min(1.0, self.manual_throttle + 0.1)
                elif keys[pygame.K_DOWN]: self.manual_throttle = max(-1.0, self.manual_throttle - 0.1)
                else: self.manual_throttle *= 0.8 # Decay to zero

                # Steering processing
                steer_limit = np.radians(23.5)
                if keys[pygame.K_LEFT]: self.manual_steering = min(steer_limit, self.manual_steering + 0.05)
                elif keys[pygame.K_RIGHT]: self.manual_steering = max(-steer_limit, self.manual_steering - 0.05)
                else: self.manual_steering *= 0.7 # Return to center

            # Run multiple simulation steps to maintain real-time speed
            for _ in range(steps_per_frame):
                if self.manual_mode:
                    throttle_pct, steering_angle = self.manual_throttle, self.manual_steering
                else:
                    throttle_pct, steering_angle = self.sim.driver.get_commands(self.sim.state, self.sim.dt)
                
                self.sim.step(throttle_pct, steering_angle, verbose=False)
                t = self.sim.t

            # Determine limit state for dynamic car coloring and HUD
            v_curr = self.sim.state['v']
            demands = self.sim.last_demands
            motor_outputs = self.sim.last_motor_outputs
            
            limit_status = "Normal"
            car_color = (0, 150, 255)  # Default Blue

            # Calculate Accelerations for G-G and Limits
            ay_ms2 = (v_curr**2 * np.tan(steering_angle) / self.sim.p['wheelbase'])
            ax_g = self.sim.state['a'] / 9.81
            ay_g = ay_ms2 / 9.81

            # Calculate Tire Data (Axle-level approximation)
            f_df = self.sim.state['f_downforce']
            delta_fz = (self.sim.mass * self.sim.state['a'] * self.sim.p['h_cg']) / self.sim.p['wheelbase']
            
            fz_r = (self.sim.mass * 9.81 * self.sim.p['weight_dist_rear']) + (f_df * self.sim.p['aero_dist_rear']) + delta_fz
            fz_f = (self.sim.mass * 9.81 * (1 - self.sim.p['weight_dist_rear'])) + (f_df * (1 - self.sim.p['aero_dist_rear'])) - delta_fz
            
            # Longitudinal forces (Propulsion - Braking)
            fx_f = (motor_outputs.get('fl_wheel_torque', 0) + motor_outputs.get('fr_wheel_torque', 0)) / self.sim.vd.R
            fx_r = motor_outputs.get('rear_axle_torque', 0) / self.sim.vd.R
            
            # Simple mechanical brake bias (matching powertrain controller logic)
            f_brake_total = demands['mechanical_brake_pct'] * (self.sim.mass * 9.81 * self.sim.p['max_brake_accel_g'])
            fx_f -= f_brake_total * (1 - self.sim.p['weight_dist_rear'])
            fx_r -= f_brake_total * self.sim.p['weight_dist_rear']

            # Lateral forces (distributed by weight)
            fy_total = self.sim.mass * ay_ms2
            tire_data = {
                "Front": {"fz": max(0, fz_f), "fx": fx_f, "fy": fy_total * (1 - self.sim.p['weight_dist_rear'])},
                "Rear":  {"fz": max(0, fz_r), "fx": fx_r, "fy": fy_total * self.sim.p['weight_dist_rear']}
            }

            # Store in history for trail rendering
            self.gg_history.append((ax_g, ay_g))

            a_lat_limit = self.sim.p['max_lat_accel_g'] * 9.81
            if abs(ay_ms2) > 0.95 * a_lat_limit:
                limit_status, car_color = "Grip Limited (Lat)", (255, 0, 0)  # Red
            elif throttle_pct > 0.95:
                if (self.sim.p['max_power_kw'] * 1000.0) / max(v_curr, 1.0) < self.sim.mass * self.sim.p['max_grip_accel_g'] * 9.81:
                    limit_status, car_color = "Power Limited", (0, 255, 0)  # Green
                else:
                    limit_status, car_color = "Grip Limited (Long)", (255, 0, 0)  # Red

            # Rendering
            self.screen.fill((10, 10, 10))

            # Draw Track Waypoints
            if self.sim.waypoints:
                for i in range(len(self.sim.waypoints)):
                    wp = self.sim.waypoints[i]
                    next_wp = self.sim.waypoints[(i + 1) % len(self.sim.waypoints)]
                    start_pos = self.world_to_screen(wp[0], wp[1])
                    end_pos = self.world_to_screen(next_wp[0], next_wp[1])
                    pygame.draw.line(self.screen, (60, 60, 60), start_pos, end_pos, 2)

            # Draw Car Position
            car_pos = self.world_to_screen(self.sim.state['x'], self.sim.state['y'])
            pygame.draw.circle(self.screen, car_color, car_pos, 6)
            
            # Draw Heading Vector
            head_x = car_pos[0] + 15 * np.cos(self.sim.state['yaw'])
            head_y = car_pos[1] - 15 * np.sin(self.sim.state['yaw'])
            pygame.draw.line(self.screen, (255, 255, 255), car_pos, (int(head_x), int(head_y)), 2)

            # Update HUD with G-units
            self.draw_hud(
                throttle = throttle_pct,
                brake=demands['mechanical_brake_pct'],
                steering=steering_angle,
                velocity=self.sim.state['v'],
                soc=motor_outputs['soc_pct'],
                temp=self.sim.battery_temp,
                p_net=motor_outputs['p_net_w'],
                limit_status=limit_status,
                ax_g=ax_g,
                ay_g=ay_g,
                tire_data=tire_data
            )

            pygame.display.flip()
            self.clock.tick(60)

            # Exit condition
            if not self.sim.running:
                print("Simulation stopped due to simulation limits.")
                pygame.time.wait(2000)
                running = False

        pygame.quit()

if __name__ == "__main__":
    viz = LiveVisualizer({
        'motor_config': 'RWD',
        'regen_enabled': True,
        'max_regen_power_kw': 20.0,
        'max_power_kw': 80.0
    })
    viz.run()