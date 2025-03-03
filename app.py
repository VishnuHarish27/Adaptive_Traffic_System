import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, session
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread, Timer
import torch
import time
from collections import deque
import json
from functools import wraps

app = Flask(__name__)
# Secret key for session
app.secret_key = 'harish@2704'

# Default credentials
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin@123!'

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return "Invalid credentials. Try again.", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Configuration
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_multi.db'
FRAME_SKIP = 5
VEHICLE_THRESHOLD = 2
MAX_QUEUE_SIZE = 32
PROCESSING_TIMES = {f'cam{i+1}': deque(maxlen=50) for i in range(4)}
IMAGE_SAVE_INTERVAL = 2  # Save images every IMAGE_SAVE_INTERVAL seconds
PROCESS_DURATION = 2  # Process each stream for 2 seconds
os.makedirs('static/temp/', exist_ok=True)

# RTSP URLs for each camera
RTSP_URLS = {
    'cam1': "rtsp://192.168.1.121:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
    'cam2': "rtsp://192.168.1.122:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
    'cam3': "rtsp://192.168.1.123:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
    'cam4': "rtsp://192.168.1.124:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true"
}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
for cam in RTSP_URLS.keys():
    os.makedirs(os.path.join(PROCESSED_FOLDER, cam), exist_ok=True)

REGIONS_FILE = 'regions_config.json'

# Hardcoded default regions with updated coordinates
DEFAULT_REGIONS = {
    'cam1': {
        'R1': {
            'vertices': np.array([(59, 378), (479, 410), (474, 185), (311, 177)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 255, 0)
        },
        'Zebra': {
            'vertices': np.array([(300,461), (2170,489)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    },
    'cam2': {
        'R1': {
            'vertices': np.array([(386, 444), (238, 302), (361, 276), (638, 385)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 255, 0)
        },
        'Zebra': {
            'vertices': np.array([(546, 484), (997, 472)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    },
    'cam3': {
        'R1': {
            'vertices': np.array([(510, 454), (67, 269), (236, 222), (640, 321)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 255, 0)
        },
        'Zebra': {
            'vertices': np.array([(436, 523), (992, 535)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    },
    'cam4': {
        'R1': {
            'vertices': np.array([(97, 476), (82, 317), (386, 279), (476, 444)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 255, 0)
        },
        'Zebra': {
            'vertices': np.array([(271, 73), (287, 694)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    }
}

# Load regions from file if it exists, and merge with defaults
if os.path.exists(REGIONS_FILE):
    with open(REGIONS_FILE, 'r') as file:
        saved_regions = json.load(file)

    for cam, regions in DEFAULT_REGIONS.items():
        if cam in saved_regions:
            for r_name, r_data in saved_regions[cam].items():
                if 'vertices' in r_data:
                    r_data['vertices'] = np.array(r_data['vertices'], dtype=np.int32)
            DEFAULT_REGIONS[cam].update(saved_regions[cam])

REGIONS = DEFAULT_REGIONS

NUM_BLOCKS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = mp.cpu_count()
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Load models
try:
    local_model_path = 'models/yolov8m.pt'
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=local_model_path,
        confidence_threshold=0.3,
        device=DEVICE
    )
    yolo_model = YOLO(local_model_path)
    yolo_model.to(DEVICE)
except Exception as e:
    print(f"Error loading models: {e}")
    detection_model = None
    yolo_model = None

def init_db(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    for cam_id in range(1, 5):
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS traffic_data_cam{cam_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                r1_vehicle_count INTEGER NOT NULL,
                r1_density REAL NOT NULL,
                weighted_vehicle_count REAL NOT NULL,
                weighted_density REAL NOT NULL,
                vdc{cam_id} BOOLEAN NOT NULL,
                processing_time REAL NOT NULL
            )
        ''')

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS zebra_crossing_data_cam{cam_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                x_position INTEGER NOT NULL,
                y_position INTEGER NOT NULL,
                road_side TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')

    conn.commit()
    conn.close()

init_db(DATABASE)

class CameraProcessor:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self.region_blocks = self.initialize_region_blocks()
        self.last_save_time = time.time()

    def initialize_region_blocks(self):
        region_data = REGIONS[f'cam{self.cam_id}']['R1']
        return self.divide_region_into_blocks(region_data['vertices'], NUM_BLOCKS)

    @staticmethod
    def is_point_in_polygon(x, y, vertices):
        return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

    @staticmethod
    def divide_region_into_blocks(vertices, num_blocks):
        min_y = min(vertices[:, 1])
        max_y = max(vertices[:, 1])
        block_height = (max_y - min_y) / num_blocks
        return [(min_y + i * block_height, min_y + (i + 1) * block_height) for i in range(num_blocks)]

    def process_frame(self, frame):
        try:
            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=True,
                postprocess_type="NMM",
                postprocess_match_threshold=0.5,
                verbose=0
            )
            return result
        except Exception as e:
            print(f"Error processing frame for camera {self.cam_id}: {e}")
            return None

    def analyze_region_detections(self, detections):
        try:
            region_data = REGIONS[f'cam{self.cam_id}']['R1']
            vertices = region_data['vertices']
            blocks = self.initialize_region_blocks()
            filled_blocks = set()
            vehicles_in_region = 0

            for pred in detections.object_prediction_list:
                if pred.category.name.lower() not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                x_center = int((x1 + x2) / 2.0)
                y_center = int((y1 + y2) / 2.0)

                if self.is_point_in_polygon(x_center, y_center, vertices):
                    vehicles_in_region += 1
                    for idx, (y_min, y_max) in enumerate(blocks):
                        if y_min <= y_center <= y_max:
                            filled_blocks.add(idx)
                            break

            density = (len(filled_blocks) / NUM_BLOCKS) * 100
            return vehicles_in_region, density, filled_blocks
        except Exception as e:
            print(f"Error analyzing region for camera {self.cam_id}: {e}")
            return 0, 0, set()

    def draw_region_visualization(self, frame, vehicles, density, filled_blocks):
        try:
            region_data = REGIONS[f'cam{self.cam_id}']['R1']
            vertices = region_data['vertices']
            color = region_data['color']
            blocks = self.initialize_region_blocks()

            cv2.polylines(frame, [vertices], isClosed=True, color=color, thickness=2)

            for idx, (y_min, y_max) in enumerate(blocks):
                block_color = color if idx in filled_blocks else (128, 128, 128)
                pts = np.array([
                    [vertices[0][0], int(y_min)],
                    [vertices[1][0], int(y_min)],
                    [vertices[1][0], int(y_max)],
                    [vertices[0][0], int(y_max)]
                ], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=block_color, thickness=1)

            text_position = (vertices[0][0], vertices[0][1] - 10)
            cv2.putText(
                frame,
                f'Vehicles: {vehicles}, Density: {density:.1f}%',
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            zebra_data = REGIONS[f'cam{self.cam_id}'].get('Zebra')
            if zebra_data:
                zebra_vertices = zebra_data['vertices']
                zebra_color = zebra_data['color']
                cv2.line(frame, tuple(zebra_vertices[0]), tuple(zebra_vertices[1]), zebra_color, 2)

        except Exception as e:
            print(f"Error visualizing region for camera {self.cam_id}: {e}")

    def process_zebra_crossing_vehicles(self, frame, detections):
        try:
            zebra_data = REGIONS.get(f'cam{self.cam_id}', {}).get('Zebra')
            if not zebra_data:
                return

            vertices = zebra_data['vertices']

            for pred in detections.object_prediction_list:
                if pred.category.name.lower() not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                x_center = int((x1 + x2) / 2.0)
                y_center = int((y1 + y2) / 2.0)

                if self.is_point_in_polygon(x_center, y_center, vertices):
                    self.save_zebra_crossing_vehicle(
                        vehicle_type=pred.category.name.lower(),
                        x_position=x_center,
                        y_position=y_center,
                        road_side='main',
                        confidence=pred.score.value
                    )

        except Exception as e:
            print(f"Error processing zebra crossing vehicles for camera {self.cam_id}: {e}")

    def save_zebra_crossing_vehicle(self, vehicle_type, x_position, y_position, road_side, confidence):
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute(f'''INSERT INTO zebra_crossing_data_cam{self.cam_id}
                               (timestamp, vehicle_type, x_position, y_position, road_side, confidence)
                               VALUES (?, ?, ?, ?, ?, ?)''',
                           (timestamp, vehicle_type, x_position, y_position, road_side, confidence))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error for camera {self.cam_id}: {e}")

    def analyze_and_save(self, frame, result):
        frame_start_time = time.time()
        processed_frame = frame.copy()
        current_time = datetime.now()

        if result is not None:
            try:
                for pred in result.object_prediction_list:
                    if pred.category.name.lower() not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    score = pred.score.value
                    category = pred.category.name

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f'{category}: {score:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + 10
                    cv2.putText(processed_frame, label, (x1, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                r1_vehicles, r1_density, r1_blocks = self.analyze_region_detections(result)
                self.draw_region_visualization(processed_frame, r1_vehicles, r1_density, r1_blocks)

                weighted_vehicles = r1_vehicles
                weighted_density = r1_density

                cv2.putText(processed_frame,
                          f'Total Vehicles: {weighted_vehicles}',
                          (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (255, 255, 255),
                          2)

                # Updated VDC logic: use the dynamic VEHICLE_THRESHOLD variable.
                vdc = 0 if r1_vehicles < VEHICLE_THRESHOLD else 1

                processing_time = time.time() - frame_start_time
                self.save_to_database(r1_vehicles, r1_density,
                                      weighted_vehicles, weighted_density,
                                      vdc, processing_time)

                if time.time() - self.last_save_time >= IMAGE_SAVE_INTERVAL:
                    filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    save_path = os.path.join(PROCESSED_FOLDER, f'cam{self.cam_id}', filename)
                    cv2.imwrite(save_path, processed_frame)
                    self.last_save_time = time.time()

                self.process_zebra_crossing_vehicles(processed_frame, result)
                return processed_frame, weighted_vehicles, weighted_density, vdc

            except Exception as e:
                print(f"Error in analyze_and_save for camera {self.cam_id}: {e}")

        return frame, 0, 0, False

    def save_to_database(self, r1_vehicles, r1_density,
                         weighted_vehicles, weighted_density, vdc, processing_time):
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute(f'''INSERT INTO traffic_data_cam{self.cam_id}
                           (timestamp, r1_vehicle_count, r1_density, 
                            weighted_vehicle_count, weighted_density, 
                            vdc{self.cam_id}, processing_time)
                           VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (timestamp, r1_vehicles, r1_density,
                         weighted_vehicles, weighted_density,
                         vdc, processing_time))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error for camera {self.cam_id}: {e}")

def process_camera_stream(cam_id, duration):
    processor = CameraProcessor(cam_id)
    cap = cv2.VideoCapture(RTSP_URLS[f'cam{cam_id}'])

    if not cap.isOpened():
        print(f"Failed to open camera {cam_id}")
        return

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        result = processor.process_frame(frame)
        processed_frame, vehicles, density, vdc = processor.analyze_and_save(frame, result)

    cap.release()
    processor.thread_pool.shutdown()

def cyclic_processing():
    while True:
        for cam_id in range(1, 5):
            process_camera_stream(cam_id, PROCESS_DURATION)
            time.sleep(0.1)

@app.route('/')
@login_required
def index():
    latest_images = {}
    for cam_id in range(1, 5):
        cam_folder = os.path.join(PROCESSED_FOLDER, f'cam{cam_id}')
        if os.path.exists(cam_folder):
            files = [f for f in os.listdir(cam_folder) if f.endswith('.jpg')]
            if files:
                latest_images[f'cam{cam_id}'] = files[-1]
            else:
                latest_images[f'cam{cam_id}'] = None
        else:
            latest_images[f'cam{cam_id}'] = None

    return render_template('index.html', latest_images=latest_images)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if request.method == 'POST':
        data = request.json
        cam_id = data.get('cam_id')
        regions = data.get('regions')

        if cam_id and regions:
            REGIONS[f'cam{cam_id}'] = regions
            return jsonify({'status': 'success', 'message': 'Regions updated'}), 200
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    return render_template('admin.html')

COLOR_MAPPING = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

@app.route('/update_regions', methods=['POST'])
def update_regions():
    try:
        data = request.json
        cam_id = data.get('cam_id')
        new_regions = data.get('regions')

        if not cam_id or not new_regions:
            return jsonify({'status': 'error', 'message': 'Missing required data'}), 400

        if cam_id not in REGIONS:
            REGIONS[cam_id] = {}

        for region in new_regions:
            region_type = region.get('type')
            vertices = region.get('vertices')
            color = region.get('color', 'green')

            if not region_type or not vertices:
                continue

            if region_type == 'Zebra':
                REGIONS[cam_id]['Zebra'] = {
                    'vertices': np.array(vertices, dtype=np.int32),
                    'color': COLOR_MAPPING.get(color.lower(), (0, 0, 255)),
                    'weight': 1.0
                }
            elif region_type == 'R':
                REGIONS[cam_id]['R1'] = {
                    'vertices': np.array(vertices, dtype=np.int32),
                    'color': COLOR_MAPPING.get(color.lower(), (0, 255, 0)),
                    'weight': 1.0
                }

        with open(REGIONS_FILE, 'w') as file:
            json_regions = {}
            for cam, regions in REGIONS.items():
                json_regions[cam] = {}
                for reg_name, reg_data in regions.items():
                    json_regions[cam][reg_name] = {
                        'vertices': reg_data['vertices'].tolist(),
                        'color': reg_data['color'],
                        'weight': reg_data['weight']
                    }
            json.dump(json_regions, file, indent=4)

        return jsonify({
            'status': 'success',
            'message': f'Regions updated successfully for {cam_id}'
        }), 200

    except Exception as e:
        print(f"Error updating regions: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error updating regions: {str(e)}'
        }), 500

@app.route('/get_regions', methods=['GET'])
def get_regions():
    try:
        return jsonify(REGIONS), 200
    except Exception as e:
        print(f"Error fetching regions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_frame/<cam_id>', methods=['GET'])
def get_frame(cam_id):
    try:
        cam_url = RTSP_URLS.get(cam_id)
        if not cam_url:
            return jsonify({'status': 'error', 'message': 'Camera not found'}), 404

        cap = cv2.VideoCapture(cam_url)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'}), 500

        filename = f'static/temp/{cam_id}_frame.jpg'
        cv2.imwrite(filename, frame)

        return jsonify({'status': 'success', 'frame_url': f'/{filename}'})
    except Exception as e:
        print(f"Error extracting frame for {cam_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/zebra_crossing_analytics')
@login_required
def zebra_crossing_analytics():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        data = {}
        for cam_id in range(1, 5):
            try:
                query = f'''
                    SELECT 
                        timestamp, vehicle_type, 
                        COUNT(*) as vehicle_count
                    FROM zebra_crossing_data_cam{cam_id}
                    GROUP BY timestamp, vehicle_type
                    ORDER BY timestamp DESC
                    LIMIT 500
                '''
                cursor.execute(query)
                rows = cursor.fetchall()

                processed_rows = [
                    {
                        'timestamp': row[0],
                        'vehicle_type': row[1],
                        'vehicle_count': row[2]
                    } for row in rows
                ]

                data[f'cam{cam_id}'] = processed_rows
            except sqlite3.Error as e:
                print(f"Error fetching zebra crossing data for camera {cam_id}: {e}")
                data[f'cam{cam_id}'] = []

        conn.close()
        return render_template('zebra_crossing.html', data=data)

    except Exception as e:
        print(f"Error in zebra crossing analytics route: {e}")
        return f"Error loading zebra crossing analytics: {str(e)}", 500

# Add these imports at the top of your Flask file
from flask import send_file
import csv
from io import StringIO
import os
from datetime import datetime, timedelta

@app.route('/analytics')
@login_required
def analytics():
    try:
        # Get date parameters
        start_date = request.args.get('start_date',
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date',
            datetime.now().strftime('%Y-%m-%d'))

        # Add one day to end_date to include the entire day
        end_date_inclusive = (datetime.strptime(end_date, '%Y-%m-%d') +
            timedelta(days=1)).strftime('%Y-%m-%d')

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        data = {}
        for cam_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT 
                    id, timestamp, r1_vehicle_count, r1_density,
                    weighted_vehicle_count, weighted_density, 
                    vdc{cam_id}, processing_time
                    FROM traffic_data_cam{cam_id} 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC LIMIT 100''',
                    (start_date, end_date_inclusive))
                rows = cursor.fetchall()
                data[f'cam{cam_id}'] = rows if rows else []
            except sqlite3.Error as e:
                print(f"Error fetching data for camera {cam_id}: {e}")
                data[f'cam{cam_id}'] = []

        conn.close()

        # Calculate aggregates
        aggregates = {}
        for cam_id in range(1, 5):
            cam_data = data[f'cam{cam_id}']
            if cam_data:
                weighted_densities = [row[4] for row in cam_data]
                vehicle_counts = [row[3] for row in cam_data]
                aggregates[f'cam{cam_id}'] = {
                    'avg_density': sum(weighted_densities) / len(weighted_densities)
                        if weighted_densities else 0,
                    'peak_count': max(vehicle_counts) if vehicle_counts else 0
                }
            else:
                aggregates[f'cam{cam_id}'] = {
                    'avg_density': 0,
                    'peak_count': 0
                }

        return render_template('analytics.html',
            data=data,
            aggregates=aggregates,
            start_date=start_date,
            end_date=end_date)

    except Exception as e:
        print(f"Error in analytics route: {e}")
        return f"Error loading analytics: {str(e)}", 500

@app.route('/download_analytics')
@login_required
def download_analytics():
    try:
        start_date = request.args.get('start_date',
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date',
            datetime.now().strftime('%Y-%m-%d'))
        end_date_inclusive = (datetime.strptime(end_date, '%Y-%m-%d') +
            timedelta(days=1)).strftime('%Y-%m-%d')

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Create a CSV string buffer
        si = StringIO()
        cw = csv.writer(si)

        # Write headers
        cw.writerow(['Camera', 'ID', 'Timestamp', 'Vehicle Count', 'Density (%)',
            'Weighted Count', 'Weighted Density (%)', 'VDC', 'Processing Time (ms)'])

        # Fetch and write data for each camera
        for cam_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT 
                    id, timestamp, r1_vehicle_count, r1_density,
                    weighted_vehicle_count, weighted_density, 
                    vdc{cam_id}, processing_time
                    FROM traffic_data_cam{cam_id} 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC''',
                    (start_date, end_date_inclusive))

                rows = cursor.fetchall()
                for row in rows:
                    cw.writerow([f'Camera {cam_id}'] + list(row))

            except sqlite3.Error as e:
                print(f"Error fetching data for camera {cam_id}: {e}")

        conn.close()

        # Create the response
        output = si.getvalue()
        si.close()

        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                f"attachment;filename=traffic_analytics_{start_date}_to_{end_date}.csv"
            }
        )

    except Exception as e:
        print(f"Error in download route: {e}")
        return f"Error downloading analytics: {str(e)}", 500

@app.route('/CameraStats', methods=['GET'])
def camera_stats():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cameras': {}
        }

        for camera_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT 
                                timestamp,
                                r1_vehicle_count,
                                r1_density,
                                weighted_vehicle_count,
                                weighted_density
                             FROM traffic_data_cam{camera_id} 
                             ORDER BY timestamp DESC LIMIT 1''')
                result = cursor.fetchone()

                if result:
                    response['cameras'][f'cam{camera_id}'] = {
                        'timestamp': result[0],
                        'vehicle_count': result[1],
                        'density': float(result[2]),
                        'total': {
                            'vehicle_count': result[3],
                            'density': float(result[4])
                        }
                    }
                else:
                    response['cameras'][f'cam{camera_id}'] = {
                        'status': 'No data available'
                    }

            except sqlite3.Error as e:
                print(f"Database error for camera {camera_id}: {e}")
                response['cameras'][f'cam{camera_id}'] = {
                    'error': str(e),
                    'status': 'Database error'
                }

        conn.close()
        return jsonify(response)

    except Exception as e:
        print(f"Server error in camera_stats: {e}")
        return jsonify({
            'error': str(e),
            'status': 'Server error',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/VehicleDetect', methods=['GET'])
def vehicle_detect():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        for camera_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT vdc{camera_id}
                             FROM traffic_data_cam{camera_id} 
                             ORDER BY timestamp DESC LIMIT 1''')
                result = cursor.fetchone()

                response[f'vdc{camera_id}'] = 1 if result and result[0] else 0

            except sqlite3.Error as e:
                print(f"Database error for camera {camera_id}: {e}")
                response[f'vdc{camera_id}'] = 0

        conn.close()
        return jsonify(response)

    except Exception as e:
        print(f"Server error in vehicle_detect: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vdc1': 0,
            'vdc2': 0,
            'vdc3': 0,
            'vdc4': 0
        }), 500

@app.route('/get_last_processed/<cam_id>')
def get_last_processed(cam_id):
    try:
        processed_folder = os.path.join(PROCESSED_FOLDER, cam_id)
        if not os.path.exists(processed_folder):
            return jsonify({'status': 'error', 'message': 'No processed images found'}), 404

        files = [f for f in os.listdir(processed_folder) if f.endswith('.jpg')]
        if not files:
            return jsonify({'status': 'error', 'message': 'No processed images found'}), 404

        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(processed_folder, x)))
        return jsonify({'status': 'success', 'image_url': f'/static/processed/{cam_id}/{latest_file}'})

    except Exception as e:
        print(f"Error getting last processed image for {cam_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_frame_skip', methods=['POST'])
def update_frame_skip():
    global FRAME_SKIP
    frame_skip = request.json.get('frame_skip')
    try:
        FRAME_SKIP = int(frame_skip)
        return jsonify({'status': 'success', 'message': 'Frame skip count updated successfully'})
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid frame skip value provided'}), 400

@app.route('/update_vehicle_threshold', methods=['POST'])
def update_vehicle_threshold():
    global VEHICLE_THRESHOLD
    threshold = request.json.get('threshold')
    try:
        VEHICLE_THRESHOLD = int(threshold)
        return jsonify({'status': 'success', 'message': 'Vehicle threshold updated successfully'})
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid threshold value provided'}), 400

@app.route('/get_parameters', methods=['GET'])
@login_required
def get_parameters():
    return jsonify({
        'frame_skip': FRAME_SKIP,
        'vehicle_threshold': VEHICLE_THRESHOLD
    })


if __name__ == "__main__":
    processing_thread = Thread(target=cyclic_processing, daemon=True)
    processing_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
