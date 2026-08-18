import pygame
import sys
import os
import numpy as np

# Ensure we can import from the simulation directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
from main import SimulationEngine

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

    def draw_hud(self, throttle, brake, steering, velocity, soc, temp, p_net):
        # Background panel for HUD
        pygame.draw.rect(self.screen, (30, 30, 30), (10, 10, 250, 200))
        pygame.draw.rect(self.screen, (100, 100, 100), (10, 10, 250, 200), 2)

        # Text Stats
        stats = [
            (f"Speed: {velocity*2.237:.1f} mph", (255, 255, 255)),
            (f"Net Power: {p_net/1000:.1f} kW", (255, 200, 0)),
            (f"Battery: {soc:.1f} %", (0, 255, 100)),
            (f"Temp: {temp:.1f} C", (255, 100, 0) if temp < 55 else (255, 0, 0))
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
        if brake > 0:
            pygame.draw.rect(self.screen, (255, 0, 0), (bar_x + 120, bar_y, int(brake * bar_w), bar_h))
        
        self.screen.blit(self.font.render("Throttle", True, (200, 200, 200)), (bar_x, bar_y - 20))
        self.screen.blit(self.font.render("Brake", True, (200, 200, 200)), (bar_x + 120, bar_y - 20))

        # Steering Visualizer
        steer_center = (1100, 100)
        pygame.draw.circle(self.screen, (100, 100, 100), steer_center, 50, 2)
        angle_rad = -steering * 5 # Exaggerate for visual clarity
        end_x = steer_center[0] + 50 * np.sin(angle_rad)
        end_y = steer_center[1] - 50 * np.cos(angle_rad)
        pygame.draw.line(self.screen, (0, 150, 255), steer_center, (end_x, end_y), 4)
        self.screen.blit(self.font.render("Steering Angle", True, (200, 200, 200)), (1050, 160))

    def run(self):
        running = True
        t = 0.0
        steps_per_frame = 2 # Simulation dt is 0.008, so 2 steps ~ 0.016s (approx 60FPS)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Run multiple simulation steps to maintain real-time speed
            for _ in range(steps_per_frame):
                throttle_pct, steering_angle = self.sim.driver.get_commands(self.sim.state, self.sim.dt)
                
                v_prev = self.sim.state['v']
                v_curr = v_prev # approximated for aero
                self.sim.state['f_downforce'] = (self.sim.p['downforce_coeff'] * v_curr**2) + self.sim.p['fan_downforce_n']
                f_drag_lookup = np.interp(v_curr, self.sim.aero_v_ms, self.sim.aero_drag_n, left=0.0)

                demands = self.sim.controller.calculate_demands(throttle_pct, steering_angle, self.sim.state, self.sim.wheel_speeds, self.sim.dt)
                motor_outputs = self.sim.powertrain.step(demands['motor_torques'], self.sim.wheel_speeds, self.sim.dt, self.sim.battery_temp)
                
                heat_removed = self.sim.thermal.calculate_heat_removed(self.sim.battery_temp, v_curr)
                self.sim.battery_temp += self.sim.thermal.get_temperature_delta(motor_outputs.get('battery_heat_w', 0.0), heat_removed, self.sim.dt)
                
                physics_in = {
                    'rear_axle_torque': motor_outputs.get('rear_axle_torque', 0),
                    'fl_wheel_torque': motor_outputs.get('fl_wheel_torque', 0),
                    'fr_wheel_torque': motor_outputs.get('fr_wheel_torque', 0),
                    'mechanical_brake_pct': demands['mechanical_brake_pct'],
                    'external_drag': f_drag_lookup
                }
                self.sim.state, self.sim.wheel_speeds = self.sim.vd.step(physics_in, steering_angle, self.sim.dt)
                t += self.sim.dt

            # Rendering
            self.screen.fill((10, 10, 10))

            # Draw Track Waypoints
            for i in range(len(self.sim.waypoints)):
                wp = self.sim.waypoints[i]
                next_wp = self.sim.waypoints[(i + 1) % len(self.sim.waypoints)]
                start_pos = self.world_to_screen(wp[0], wp[1])
                end_pos = self.world_to_screen(next_wp[0], next_wp[1])
                pygame.draw.line(self.screen, (60, 60, 60), start_pos, end_pos, 2)

            # Draw Car Position
            car_pos = self.world_to_screen(self.sim.state['x'], self.sim.state['y'])
            pygame.draw.circle(self.screen, (0, 150, 255), car_pos, 6)
            
            # Draw Heading Vector
            head_x = car_pos[0] + 15 * np.cos(self.sim.state['yaw'])
            head_y = car_pos[1] - 15 * np.sin(self.sim.state['yaw'])
            pygame.draw.line(self.screen, (255, 255, 255), car_pos, (int(head_x), int(head_y)), 2)

            # Update HUD
            self.draw_hud(
                throttle=max(0, throttle_pct),
                brake=demands['mechanical_brake_pct'],
                steering=steering_angle,
                velocity=self.sim.state['v'],
                soc=motor_outputs['soc_pct'],
                temp=self.sim.battery_temp,
                p_net=motor_outputs['p_net_w']
            )

            pygame.display.flip()
            self.clock.tick(60)

            # Exit condition
            if self.sim.battery_temp > 60.0 or motor_outputs['soe_pct'] <= 0.0:
                print("Simulation stopped due to thermal/energy limits.")
                pygame.time.wait(2000)
                running = False

        pygame.quit()

if __name__ == "__main__":
    viz = LiveVisualizer({
        'motor_config': 'RWD',
        'regen_enabled': True,
        'max_regen_power_kw': 20.0
    })
    viz.run()