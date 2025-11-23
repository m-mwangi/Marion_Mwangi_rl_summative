import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import time
import numpy as np

class FireRescueVisualizer:
    def __init__(self, env):
        self.env = env
        self.window_width = 1280
        self.window_height = 720
        self.camera_distance = 20
        self.camera_angle = 45
        self.camera_height = 15

        self.last_time = 0
        self.fps = 0
        self.frame_count = 0

        self.wheel_rotation = 0
        self.fixed_camera = True
        
        # Fire visual effects
        self.fire_pulse = 0
        self.fire_timer = 0
        
        # Smoke / dust behind robot
        self.smoke_particles = []

        # Mission status banner
        self.status_message = ""
        self.status_timer = 0

        # Robot water-transfer animation
        self.water_transition = 0.0
        
        self._init_opengl()

    #  OpenGL Setup
    def _init_opengl(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow(b"Fire Rescue Simulator")

        glutDisplayFunc(self._render)
        glutIdleFunc(self._update)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutSpecialFunc(self._special_keys)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)

        # Fire glow spotlight
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 0.3, 0.0, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [1.0, 0.2, 0.0, 1])
        glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 35.0)
        glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 2.0)

        # Better visuals
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearDepth(1.0)
        glDepthFunc(GL_LEQUAL)

    def _reshape(self, width, height):
        self.window_width = width
        self.window_height = height
        glViewport(0, 0, width, height)
        self._setup_camera()

    #  Controls
    def _keyboard(self, key, x, y):
        if key == b'\x1b':
            glutDestroyWindow(glutGetWindow())
        elif key in (b'f', b'F'):
            self.fixed_camera = not self.fixed_camera

    def _special_keys(self, key, x, y):
        mod = glutGetModifiers()
        if mod & GLUT_ACTIVE_CTRL:
            if key == GLUT_KEY_UP:
                self.camera_distance = max(10, self.camera_distance - 1)
            elif key == GLUT_KEY_DOWN:
                self.camera_distance = min(50, self.camera_distance + 1)
        elif key == GLUT_KEY_LEFT:
            self.camera_angle = (self.camera_angle + 5) % 360
        elif key == GLUT_KEY_RIGHT:
            self.camera_angle = (self.camera_angle - 5) % 360

    #  Update Loop
    def _update(self):
        current_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0
        dt = current_time - self.last_time

        # FPS counter
        self.frame_count += 1
        if dt >= 1.0:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.last_time = current_time

        # Fire pulsing animation
        self.fire_timer += dt
        self.fire_pulse = abs(math.sin(self.fire_timer * 3))

        # Wheel rotation
        speed = self.env.current_speed
        wheel_circ = 2.0
        rot_per_sec = speed / wheel_circ
        deg_per_sec = rot_per_sec * 360
        self.wheel_rotation = (self.wheel_rotation + deg_per_sec * dt) % 360

        # Smoke particles
        self._update_smoke(dt)

        glutPostRedisplay()

    #  Particle Effects
    def _update_smoke(self, dt):
        new_particles = []
        for p in self.smoke_particles:
            p["pos"][0] += p["vel"][0] * dt
            p["pos"][1] += p["vel"][1] * dt
            p["pos"][2] += p["vel"][2] * dt
            p["life"] -= dt
            if p["life"] > 0:
                new_particles.append(p)
        self.smoke_particles = new_particles

        # Emit dust during fast turns or high acceleration
        if self.env.current_speed > 5.5:
            self._add_dust_clouds()

    def _add_dust_clouds(self):
        for _ in range(4):
            self.smoke_particles.append({
                "pos": [
                    -0.4 + random.uniform(-0.1, 0.1),
                    random.choice([-0.25, 0.25]),
                    -0.1
                ],
                "vel": [
                    random.uniform(-1.0, -0.5),
                    random.uniform(-0.3, 0.3),
                    random.uniform(0.1, 0.4)
                ],
                "size": random.uniform(0.12, 0.18),
                "life": random.uniform(0.4, 1.0),
                "color": [0.2, 0.2, 0.2, 0.7]
            })

    #  Camera
    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.window_width/self.window_height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    #  Render Frame
    def render(self):
        glutMainLoopEvent()
        self._render()

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera position
        if self.fixed_camera:
            cx = self.env.grid_size[0] / 2
            cy = self.env.grid_size[1] / 2
            gluLookAt(cx, cy, self.camera_height,
                      cx, cy, 0,
                      0, 1, 0)
        else:
            rx, ry = self.env.robot_pos
            cam_x = rx - math.sin(math.radians(self.camera_angle)) * self.camera_distance
            cam_y = ry - math.cos(math.radians(self.camera_angle)) * self.camera_distance
            gluLookAt(cam_x, cam_y, self.camera_height,
                      rx, ry, 0,
                      0, 0, 1)

        # Lighting for fire glow
        nearest_fire = self._nearest_fire_light()
        if nearest_fire is not None:
            glEnable(GL_LIGHT1)
            glLightfv(GL_LIGHT1, GL_POSITION, [nearest_fire[0], nearest_fire[1], 2.5, 1])
        else:
            glDisable(GL_LIGHT1)

        # World Elements
        self._draw_ground()
        self._draw_grid()
        self._draw_debris()
        self._draw_water_stations()
        self._draw_fires()
        self._draw_robot()
        self._draw_smoke()

        glutSwapBuffers()

    #  Drawing Functions
    def _draw_ground(self):
        glColor3f(0.25, 0.45, 0.25)
        minx, maxx = self.env.min_x, self.env.max_x
        miny, maxy = self.env.min_y, self.env.max_y

        glBegin(GL_QUADS)
        glVertex3f(minx-2, miny-2, -0.1)
        glVertex3f(maxx+2, miny-2, -0.1)
        glVertex3f(maxx+2, maxy+2, -0.1)
        glVertex3f(minx-2, maxy+2, -0.1)
        glEnd()

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor4f(1, 1, 1, 0.4)
        for y in np.arange(0, self.env.grid_size[1]+1, 1):
            glBegin(GL_LINES)
            glVertex3f(0, y, 0)
            glVertex3f(self.env.grid_size[0], y, 0)
            glEnd()
        for x in np.arange(0, self.env.grid_size[0]+1, 1):
            glBegin(GL_LINES)
            glVertex3f(x, 0, 0)
            glVertex3f(x, self.env.grid_size[1], 0)
            glEnd()
        glEnable(GL_LIGHTING)

    # Fires
    def _draw_fires(self):
        for f in self.env.active_fires:
            self._draw_fire(f)

    def _draw_fire(self, pos):
        x, y = pos
        glPushMatrix()
        glTranslatef(x, y, 0.3)

        pulse = 1 + self.fire_pulse * 0.4
        glColor3f(1.0, 0.3, 0.0)
        glScalef(0.4*pulse, 0.4*pulse, 0.6*pulse)
        glutSolidSphere(0.4, 16, 16)

        glPopMatrix()

    def _nearest_fire_light(self):
        if not self.env.active_fires:
            return None
        robot = self.env.robot_pos
        distances = [np.linalg.norm(robot - f) for f in self.env.active_fires]
        idx = np.argmin(distances)
        return self.env.active_fires[idx]

    # Water Stations
    def _draw_water_stations(self):
        for ws in self.env.water_sources:
            self._draw_water_station(ws)

    def _draw_water_station(self, pos):
        x, y = pos
        glPushMatrix()
        glTranslatef(x, y, 0)

        glColor3f(0.2, 0.4, 1.0)
        glScalef(0.8, 0.8, 0.4)
        glutSolidCube(1)

        glColor4f(0.7, 0.8, 1.0, 0.8)
        glTranslatef(0, 0, 0.6)
        glScalef(0.6, 0.6, 0.2)
        glutSolidCube(1)

        glPopMatrix()

    # Debris (Obstacles)
    def _draw_debris(self):
        for d in self.env.obstacles:
            x, y = d
            glPushMatrix()
            glTranslatef(x, y, 0.2)

            glColor3f(0.5, 0.3, 0.1)
            glScalef(0.9, 0.9, 0.4)
            glutSolidCube(1)
            glPopMatrix()

    # Robot
    def _draw_robot(self):
        x, y = self.env.robot_pos
        dx, dy = self.env.robot_dir

        rot = math.degrees(math.atan2(dy, dx))

        glPushMatrix()
        glTranslatef(x, y, 0.3)
        glRotatef(rot, 0, 0, 1)

        # Base body
        glColor3f(0.7, 0.1, 0.1)
        glPushMatrix()
        glScalef(0.9, 0.5, 0.4)
        glutSolidCube(1)
        glPopMatrix()

        # Sensor dome
        glColor3f(0.8, 0.8, 0.8)
        glPushMatrix()
        glTranslatef(0.2, 0, 0.35)
        glutSolidSphere(0.18, 16, 16)
        glPopMatrix()

        # Water tank indicator
        lvl = self.env.water_level / self.env.max_water
        glDisable(GL_LIGHTING)
        glColor3f(0, 0.6, 1)
        glBegin(GL_QUADS)
        glVertex3f(-0.3, 0.3, 0.45)
        glVertex3f(-0.3 + 0.6 * lvl, 0.3, 0.45)
        glVertex3f(-0.3 + 0.6 * lvl, 0.35, 0.45)
        glVertex3f(-0.3, 0.35, 0.45)
        glEnd()
        glEnable(GL_LIGHTING)

        # Wheels
        for wx, wy in [(0.35,0.25),(0.35,-0.25),(-0.35,0.25),(-0.35,-0.25)]:
            glPushMatrix()
            glTranslatef(wx, wy, -0.2)
            glRotatef(90, 0, 1, 0)
            glRotatef(self.wheel_rotation, 0, 0, 1)
            glutSolidTorus(0.05, 0.1, 8, 16)
            glPopMatrix()

        glPopMatrix()

    # Smoke
    def _draw_smoke(self):
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)

        for p in self.smoke_particles:
            r, g, b, a = p["color"]
            alpha = a * (p["life"] / 1.0)
            glColor4f(r, g, b, alpha)

            glPushMatrix()
            glTranslatef(*p["pos"])
            glutSolidSphere(p["size"], 8, 8)
            glPopMatrix()

        glDepthMask(True)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def close(self):
        glutDestroyWindow(glutGetWindow())
