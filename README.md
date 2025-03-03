# ATCS Project 

## 1. Project Overview

The Automated Traffic Control System (ATCS) is designed to monitor and analyze traffic using camera feeds. It processes video streams to detect vehicles, analyze traffic patterns, and provide visualizations and analytics. The system is built using Python and Flask for the backend, OpenCV for image processing, and various machine learning models for detecting vehicles and analyzing traffic.

## 2. Setup Instructions

### Prerequisites

To set up the ATCS project, ensure you have the following prerequisites installed:
- Docker: A platform to develop, ship, and run applications inside containers.
- Python 3.8.8: The programming language used for the project.

### Installation Steps

1. **Clone the Repository**
   Clone the ATCS repository from GitHub to your local machine using the following command:
   ```
   git clone https://github.com/HarishVishnu27/ATCS.git
   cd ATCS
   ```

2. **Build the Docker Image**
   Build the Docker image for the project using the following command:
   ```
   docker build -t atcs .
   ```

3. **Run the Docker Container**
   Run the Docker container for the project using the following command:
   ```
   docker run -p 5000:5000 atcs
   ```

### Dependencies

The project dependencies are listed in the `requirements.txt` file. These include Flask for the web framework, NumPy for numerical operations, OpenCV for image processing, and various machine learning libraries like Ultralytics and Torch.

## 3. File Descriptions

### Dockerfile
The `Dockerfile` sets up the environment for the application. It uses the Python 3.8.8-slim base image, sets the working directory, copies the application files, installs the dependencies, exposes the application port, and sets the command to run the application.

### README.md
The `README.md` file provides a brief overview of the project, including its purpose and basic setup instructions.

### app.py
The `app.py` file is the main application file. It initializes the Flask app, configures the application settings, defines routes for different functionalities, and implements the `CameraProcessor` class for processing camera streams and analyzing traffic.

### regions_config.json
The `regions_config.json` file contains the configuration for different camera regions, including vertices, colors, and weights. It is used to define and visualize regions in the camera feeds.

### requirements.txt
The `requirements.txt` file lists all the dependencies required for the project. These include Flask, NumPy, OpenCV, Ultralytics, and more.

### static/styles.css
The `styles.css` file contains the CSS styles for the web application, including styles for the header, navigation, main content, footer, and various UI components.

### HTML Templates

#### templates/admin.html
The `admin.html` file provides the admin panel interface. It allows updating system parameters like frame skip count and vehicle detection threshold, selecting and configuring camera regions and zebra crossings, displaying live camera feed, and showing the last processed image for reference.

#### templates/analytics.html
The `analytics.html` file displays traffic analytics. It includes features like date filtering, CSV download, summary cards for average vehicle density and peak vehicle count, charts for weighted density over time, and detailed data tables for each camera.

#### templates/index.html
The `index.html` file is the main dashboard interface. It displays the latest images from each camera, shows vehicle count and density, indicates VDC (Vehicle Density Control) status, and updates the camera feeds and statistics periodically.

#### templates/login.html
The `login.html` file provides the login interface for the application. It includes a form for username and password authentication.

#### templates/zebra_crossing.html
The `zebra_crossing.html` file displays analytics for zebra crossings detected by each camera. It allows filtering data by vehicle type and date, and shows detailed tables with information about vehicles detected at zebra crossings.

## 4. How the Project Works

### Initialization
- The Flask application is initialized in `app.py`.
- Default configurations and settings are loaded, including database settings, processing intervals, RTSP URLs for camera feeds, and region configurations.

### Processing Camera Streams
- The `CameraProcessor` class is responsible for processing camera streams.
- It initializes region blocks, processes frames, analyzes detections, and saves relevant data to the database.
- Functions like `process_frame`, `analyze_region_detections`, and `save_zebra_crossing_vehicle` handle different aspects of traffic analysis.

### Routes and Endpoints
- The application defines several routes for handling different functionalities:
  - `/update_regions`: Updates the regions configuration.
  - `/get_regions`: Retrieves the current regions configuration.
  - `/get_frame`: Fetches a frame from a specified camera.
  - `/analytics`: Provides traffic analytics.
  - `/camera_stats`: Retrieves camera statistics.
  - `/vehicle_detect`: Detects vehicles in the camera feed.
  - `/get_last_processed`: Retrieves the last processed frame from a camera.
  - `/update_frame_skip`: Updates the frame skip setting.
  - `/update_vehicle_threshold`: Updates the vehicle detection threshold.
  - `/get_parameters`: Retrieves the current application parameters.

### Visualization and Analytics
- The application provides visualizations for traffic analysis, including vehicle detections and density.
- Uses OpenCV and Matplotlib for processing and visualizing the data.

## 5. Running the Project

### Start the Docker Container
Run the Docker container using the following command:
```
docker run -p 5000:5000 atcs
```

### Access the Web Application
Open a web browser and navigate to `http://localhost:5000`. Use the default credentials (username: `admin`, password: `admin@123!`) to log in.

### Upload and Configure Regions
Use the `/update_regions` endpoint to upload and configure camera regions. Visualize the configured regions to ensure they match the camera feeds.

### Monitor Traffic
The application will start processing the camera feeds and provide real-time traffic analysis. Access various analytics and visualizations through the web interface.

### Functionalities of Each Page

#### Admin Panel (`templates/admin.html`)
- **System Parameters**: Allows updating frame skip count and vehicle detection threshold.
- **Camera Selection**: Enables selection of which camera to configure.
- **Region Selection**: Allows selecting the type of region (Traffic Region or Zebra Crossing) to draw.
- **Live Camera Feed**: Displays live feed from the selected camera for drawing regions.
- **Last Processed Image**: Shows the last processed image for reference.
- **Save and Clear Regions**: Provides options to save the drawn regions to the configuration or clear the drawing area.

#### Analytics Page (`templates/analytics.html`)
- **Date Filter**: Allows filtering traffic data by date range.
- **Summary Cards**: Displays average vehicle density and peak vehicle count for each camera.
- **Charts**: Shows weighted density over time for each camera.
- **Data Tables**: Displays detailed traffic data in tabular format for each camera.
- **Download CSV**: Provides an option to download the filtered traffic data as a CSV file.

#### Dashboard (`templates/index.html`)
- **Camera Feeds**: Displays the latest images from each camera.
- **Vehicle Count and Density**: Shows the current vehicle count and density for each camera.
- **VDC Status**: Indicates whether the Vehicle Density Control (VDC) is active.
- **Periodic Updates**: Updates the camera feeds and statistics every second.

#### Login Page (`templates/login.html`)
- **Login Form**: Provides a form for username and password authentication.
- **Background Image**: Displays a background image for the login page.

#### Zebra Crossing Analytics (`templates/zebra_crossing.html`)
- **Vehicle Type Filter**: Allows filtering zebra crossing data by vehicle type.
- **Date Filter**: Allows filtering zebra crossing data by date.
- **Data Tables**: Displays detailed zebra crossing data in tabular format for each camera.
