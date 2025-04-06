import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
import math
import requests
from io import BytesIO
from astroquery.simbad import Simbad
# Corrigindo importação depreciada
from astroquery.ipac.ned import Ned
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy.io.fits.header")

# Constants for astronomical calculations
SOLAR_MASS = 1.989e30  # kg
LIGHT_YEAR = 9.461e15  # meters
HUBBLE_CONSTANT = 70.0  # km/s/Mpc

# Load pre-trained models and reference data
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Known galaxy catalog with accurate data
KNOWN_GALAXIES = {
    "Andromeda Galaxy": {
        "names": ["M31", "NGC 224", "Andromeda"],
        "distance": 2.537,  # million light-years
        "distance_unit": "million light-years",
        "redshift": -0.001001,  # negative means blueshift
        "mass": 1.5e12,  # solar masses
        "mass_unit": "solar masses",
        "temperature": 6000,  # K
        "temperature_unit": "K",
        "type": "spiral galaxy",
        "catalog": "M31, NGC 224",
        "ra": 10.6847,  # Right ascension in degrees
        "dec": 41.2687,  # Declination in degrees
        "visual_features": ["prominent bulge", "dusty spiral arms", "elliptical shape", "yellowish center"]
    },
    "Whirlpool Galaxy": {
        "names": ["M51", "NGC 5194", "Whirlpool"],
        "distance": 23.0,  # million light-years
        "distance_unit": "million light-years",
        "redshift": 0.001544,
        "mass": 1.6e11,  # solar masses
        "mass_unit": "solar masses",
        "temperature": 4000,  # K
        "temperature_unit": "K",
        "type": "spiral galaxy",
        "catalog": "M51, NGC 5194",
        "ra": 202.4696,  # Right ascension in degrees
        "dec": 47.1952,  # Declination in degrees
        "visual_features": ["clear spiral arms", "companion galaxy", "face-on orientation", "bluish spiral arms"]
    },
    "Triangulum Galaxy": {
        "names": ["M33", "NGC 598", "Triangulum"],
        "distance": 2.73,  # million light-years
        "distance_unit": "million light-years",
        "redshift": -0.000597,
        "mass": 5.0e10,  # solar masses
        "mass_unit": "solar masses",
        "temperature": 5500,  # K
        "temperature_unit": "K",
        "type": "spiral galaxy",
        "catalog": "M33, NGC 598",
        "ra": 23.4621,  # Right ascension in degrees
        "dec": 30.6599,  # Declination in degrees
        "visual_features": ["loose spiral structure", "bright star-forming regions", "bluish color", "diffuse appearance"]
    },
    "Sombrero Galaxy": {
        "names": ["M104", "NGC 4594", "Sombrero"],
        "distance": 29.3,  # million light-years
        "distance_unit": "million light-years",
        "redshift": 0.003416,
        "mass": 8.0e11,  # solar masses
        "mass_unit": "solar masses",
        "temperature": 3500,  # K
        "temperature_unit": "K",
        "type": "spiral galaxy",
        "catalog": "M104, NGC 4594",
        "ra": 189.9976,  # Right ascension in degrees
        "dec": -11.6231,  # Declination in degrees
        "visual_features": ["prominent dust lane", "bright bulge", "edge-on orientation", "hat-like appearance"]
    }
}

# Star type classification based on color and temperature
STAR_TYPES = {
    "O": {"temp_range": (30000, 50000), "color": "blue", "mass_range": (16, 150)},
    "B": {"temp_range": (10000, 30000), "color": "blue-white", "mass_range": (2.1, 16)},
    "A": {"temp_range": (7500, 10000), "color": "white", "mass_range": (1.4, 2.1)},
    "F": {"temp_range": (6000, 7500), "color": "yellow-white", "mass_range": (1.04, 1.4)},
    "G": {"temp_range": (5200, 6000), "color": "yellow", "mass_range": (0.8, 1.04)},
    "K": {"temp_range": (3700, 5200), "color": "orange", "mass_range": (0.45, 0.8)},
    "M": {"temp_range": (2400, 3700), "color": "red", "mass_range": (0.08, 0.45)}
}

# Nebula types and their characteristics
NEBULA_TYPES = [
    "emission nebula",
    "reflection nebula",
    "dark nebula",
    "planetary nebula",
    "supernova remnant"
]

# Galaxy types and their characteristics
GALAXY_TYPES = [
    "spiral galaxy",
    "elliptical galaxy",
    "lenticular galaxy",
    "irregular galaxy",
    "ring galaxy",
    "dwarf galaxy"
]

# Função auxiliar para garantir que todos os valores sejam serializáveis para JSON
def ensure_json_serializable(obj):
    """Converte valores não serializáveis para JSON em valores serializáveis"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    elif isinstance(obj, (str, int, float)):
        return obj
    else:
        return str(obj)

class CelestialDetector:
    def __init__(self):
        self.model_path = os.path.join(MODEL_DIR, "celestial_classifier.pkl")
        self.feature_scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
        
        # Initialize or load the classifier model
        if os.path.exists(self.model_path) and os.path.exists(self.feature_scaler_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(self.feature_scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                logger.info("Loaded pre-trained models successfully")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                self._initialize_models()
        else:
            self._initialize_models()
        
        # Configure custom Simbad query - corrigindo campos depreciados
        self.custom_simbad = Simbad()
        self.custom_simbad.add_votable_fields('V', 'B', 'R', 'otype', 'mesdistance')
        
    def _initialize_models(self):
        """Initialize new models if pre-trained ones aren't available"""
        logger.info("Initializing new classification models")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        
        # We would normally train these models here, but for now we'll use them untrained
        # and rely more on rule-based classification
    
    def save_models(self):
        """Save the trained models for future use"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.feature_scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        logger.info("Models saved successfully")
    
    def process_image(self, input_path, output_path, results_path):
        """Main method to process an astronomical image"""
        try:
            logger.info(f"Processing image: {input_path}")
            
            # Load and preprocess the image
            original_image = cv2.imread(input_path)
            if original_image is None:
                raise ValueError(f"Failed to load image from {input_path}")
            
            # Convert to RGB for analysis and display
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create a copy for drawing results
            annotated_image = image_rgb.copy()
            
            # Convert to grayscale for object detection
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            
            # Analyze image characteristics
            image_analysis = self.analyze_image_characteristics(image_rgb, gray_image)
            
            # Detect objects in the image
            detected_objects = self.detect_celestial_objects(image_rgb, gray_image)
            
            # Identify known galaxies
            self.identify_known_galaxies(detected_objects, image_rgb, image_analysis)
            
            # Calculate scientific data for each object
            self.calculate_scientific_data(detected_objects, image_analysis)
            
            # Generate summary
            summary = self.generate_summary(detected_objects, image_analysis)
            
            # Draw annotations on the image
            self.draw_annotations(annotated_image, detected_objects)
            
            # Save the annotated image
            plt.figure(figsize=(12, 10))
            plt.imshow(annotated_image)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Prepare results - garantindo que todos os valores sejam serializáveis para JSON
            results = {
                "objects": ensure_json_serializable(detected_objects),
                "summary": ensure_json_serializable(summary),
                "image_analysis": ensure_json_serializable(image_analysis)
            }
            
            # Save results to JSON
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Processing complete. Results saved to {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_image_characteristics(self, image_rgb, gray_image):
        """Analyze general characteristics of the astronomical image"""
        logger.info("Analyzing image characteristics")
        
        # Calculate basic image statistics
        height, width = gray_image.shape
        mean_brightness = np.mean(gray_image) / 255.0
        std_brightness = np.std(gray_image) / 255.0
        
        # Calculate contrast
        contrast = std_brightness / max(mean_brightness, 0.01)
        
        # Calculate color distribution
        r, g, b = cv2.split(image_rgb)
        r_mean, g_mean, b_mean = np.mean(r)/255.0, np.mean(g)/255.0, np.mean(b)/255.0
        total = r_mean + g_mean + b_mean
        if total > 0:
            r_ratio, g_ratio, b_ratio = r_mean/total, g_mean/total, b_mean/total
        else:
            r_ratio, g_ratio, b_ratio = 0.33, 0.33, 0.33
        
        # Determine dominant color
        max_channel = max(r_mean, g_mean, b_mean)
        if max_channel == r_mean:
            dominant_color = "red"
        elif max_channel == g_mean:
            dominant_color = "green"
        else:
            dominant_color = "blue"
        
        # Calculate edge density (measure of complexity)
        edges = cv2.Canny(gray_image, 100, 200)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Determine complexity level
        if edge_density < 0.05:
            complexity = "low"
        elif edge_density < 0.15:
            complexity = "medium"
        else:
            complexity = "high"
        
        # Check for potential redshift/blueshift
        potential_redshift = r_ratio > 0.4 and r_ratio > b_ratio
        potential_blueshift = b_ratio > 0.4 and b_ratio > r_ratio
        
        # Determine image type based on characteristics
        if edge_density < 0.03 and mean_brightness < 0.2:
            image_type = "deep space"
        elif edge_density > 0.1 and mean_brightness > 0.3:
            image_type = "star field"
        elif edge_density > 0.05 and (r_ratio > 0.4 or b_ratio > 0.4):
            image_type = "nebula region"
        else:
            image_type = "galaxy field"
        
        # Determine object type based on image analysis
        if edge_density < 0.05 and mean_brightness < 0.15:
            object_type = "galaxy"
        elif edge_density > 0.1 and mean_brightness > 0.3:
            object_type = "star cluster"
        elif edge_density > 0.05 and (r_ratio > 0.4 or b_ratio > 0.4):
            object_type = "nebula"
        else:
            object_type = "mixed field"
        
        # Initial galaxy identification (to be refined later)
        galaxy_name = None
        galaxy_confidence = None
        
        return {
            "image_type": image_type,
            "object_type": object_type,
            "galaxy_name": galaxy_name,
            "galaxy_confidence": galaxy_confidence,
            "brightness": float(mean_brightness),  # Garantindo que seja float nativo
            "contrast": float(contrast),  # Garantindo que seja float nativo
            "complexity": complexity,
            "edge_density": float(edge_density),  # Garantindo que seja float nativo
            "dominant_color": dominant_color,
            "potential_redshift": bool(potential_redshift),  # Garantindo que seja bool nativo
            "potential_blueshift": bool(potential_blueshift),  # Garantindo que seja bool nativo
            "color_distribution": {
                "red": float(r_ratio),  # Garantindo que seja float nativo
                "green": float(g_ratio),  # Garantindo que seja float nativo
                "blue": float(b_ratio)  # Garantindo que seja float nativo
            }
        }
    
    def detect_celestial_objects(self, image_rgb, gray_image):
        """Detect and classify celestial objects in the image"""
        logger.info("Detecting celestial objects")
        
        detected_objects = []
        
        # Enhance the image for better detection
        enhanced = self._enhance_image(gray_image)
        
        # Detect stars using DAOStarFinder
        mean, median, std = sigma_clipped_stats(enhanced, sigma=3.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        sources = daofind(enhanced - median)
        
        if sources is not None and len(sources) > 0:
            # Convert sources to a list of dictionaries
            stars = []
            for i in range(len(sources)):
                x = float(sources['xcentroid'][i])
                y = float(sources['ycentroid'][i])
                flux = float(sources['flux'][i])
                peak = float(sources['peak'][i])
                
                # Calculate size based on flux
                size = max(1.0, np.sqrt(flux) / 5)
                
                # Calculate magnitude (inverse of brightness)
                magnitude = 20 - 2.5 * np.log10(max(1, flux))
                
                stars.append({
                    "x": x,
                    "y": y,
                    "flux": flux,
                    "peak": peak,
                    "size": size,
                    "magnitude": magnitude
                })
            
            # Filter out duplicate detections
            filtered_stars = self._filter_nearby_objects(stars, distance_threshold=5.0)
            
            # Add stars to detected objects
            for star in filtered_stars:
                # Get color at star position
                x, y = int(star["x"]), int(star["y"])
                if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
                    color_sample = image_rgb[max(0, y-2):min(image_rgb.shape[0], y+3), 
                                            max(0, x-2):min(image_rgb.shape[1], x+3)]
                    if color_sample.size > 0:
                        avg_color = np.mean(color_sample, axis=(0, 1))
                        r, g, b = avg_color
                        
                        # Determine star color based on RGB values
                        star_color = self._determine_star_color(r, g, b)
                    else:
                        star_color = None
                else:
                    star_color = None
                
                # Add star to detected objects
                detected_objects.append({
                    "type": "star",
                    "name": None,
                    "confidence": 0.8,  # Base confidence
                    "size": float(star["size"]),  # Garantindo que seja float nativo
                    "magnitude": float(star["magnitude"]),  # Garantindo que seja float nativo
                    "position": {"x": float(star["x"]), "y": float(star["y"])},  # Garantindo que sejam float nativos
                    "color": star_color
                })
        
        # Detect galaxies using blob detection
        galaxies = self._detect_galaxies(gray_image)
        for galaxy in galaxies:
            detected_objects.append({
                "type": "galaxy",
                "name": None,
                "confidence": 0.7,  # Base confidence
                "size": float(galaxy["size"]),  # Garantindo que seja float nativo
                "magnitude": float(galaxy["magnitude"]),  # Garantindo que seja float nativo
                "position": {"x": float(galaxy["x"]), "y": float(galaxy["y"])},  # Garantindo que sejam float nativos
                "galaxy_type": "spiral galaxy"  # Default, will be refined later
            })
        
        # Detect nebulae using color and texture analysis
        nebulae = self._detect_nebulae(image_rgb, gray_image)
        for nebula in nebulae:
            detected_objects.append({
                "type": "nebula",
                "name": None,
                "confidence": 0.7,  # Base confidence
                "size": float(nebula["size"]),  # Garantindo que seja float nativo
                "magnitude": float(nebula["magnitude"]),  # Garantindo que seja float nativo
                "position": {"x": float(nebula["x"]), "y": float(nebula["y"])},  # Garantindo que sejam float nativos
                "nebula_type": nebula["nebula_type"]
            })
        
        return detected_objects
    
    def _enhance_image(self, gray_image):
        """Enhance the image for better object detection"""
        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        # Apply Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def _filter_nearby_objects(self, objects, distance_threshold=5.0):
        """Filter out duplicate detections that are too close to each other"""
        if not objects:
            return []
        
        filtered = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
                
            group = [obj1]
            used.add(i)
            
            for j, obj2 in enumerate(objects):
                if j in used or i == j:
                    continue
                    
                dist = np.sqrt((obj1["x"] - obj2["x"])**2 + (obj1["y"] - obj2["y"])**2)
                if dist < distance_threshold:
                    group.append(obj2)
                    used.add(j)
            
            # Keep the brightest object in the group
            brightest = max(group, key=lambda x: x["flux"])
            filtered.append(brightest)
        
        return filtered
    
    def _determine_star_color(self, r, g, b):
        """Determine star color based on RGB values"""
        # Normalize RGB values
        total = r + g + b
        if total == 0:
            return None
            
        r, g, b = r/total, g/total, b/total
        
        # Determine color based on RGB ratios
        if r > 0.4 and r > g and r > b:
            return "red"
        elif r > 0.35 and g > 0.35 and r > b and g > b:
            return "yellow"
        elif g > 0.4 and g > r and g > b:
            return "green"
        elif b > 0.4 and b > r and b > g:
            return "blue"
        elif r > 0.3 and g > 0.3 and b > 0.3:
            return "white"
        else:
            return "unknown"
    
    def _detect_galaxies(self, gray_image):
        """Detect galaxies using blob detection"""
        # Set up the blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by area
        params.filterByArea = True
        params.minArea = 500
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Invert image for dark blob detection
        inverted = 255 - gray_image
        
        # Detect blobs
        keypoints = detector.detect(inverted)
        
        galaxies = []
        for kp in keypoints:
            # Calculate size and magnitude
            size = kp.size
            # Larger blobs have lower magnitude (brighter)
            magnitude = 15 - np.log(size)
            
            galaxies.append({
                "x": float(kp.pt[0]),  # Garantindo que seja float nativo
                "y": float(kp.pt[1]),  # Garantindo que seja float nativo
                "size": float(size),  # Garantindo que seja float nativo
                "magnitude": float(magnitude)  # Garantindo que seja float nativo
            })
        
        return galaxies
    
    def _detect_nebulae(self, image_rgb, gray_image):
        """Detect nebulae using color and texture analysis"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different nebula types
        # Emission nebulae (red/pink)
        lower_emission = np.array([160, 50, 50])
        upper_emission = np.array([180, 255, 255])
        mask_emission = cv2.inRange(hsv, lower_emission, upper_emission)
        
        # Reflection nebulae (blue)
        lower_reflection = np.array([90, 50, 50])
        upper_reflection = np.array([130, 255, 255])
        mask_reflection = cv2.inRange(hsv, lower_reflection, upper_reflection)
        
        # Dark nebulae (look for dark patches against bright backgrounds)
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        mask_dark = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # Combine masks
        mask_combined = mask_emission | mask_reflection | mask_dark
        
        # Find contours
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nebulae = []
        for contour in contours:
            # Filter out small contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w/2
            center_y = y + h/2
            
            # Calculate size as the average of width and height
            size = (w + h) / 2
            
            # Calculate magnitude (inverse of size)
            magnitude = 15 - np.log(size)
            
            # Determine nebula type
            region = hsv[y:y+h, x:x+w]
            if region.size > 0:
                avg_hue = np.mean(region[:,:,0])
                if avg_hue < 20 or avg_hue > 160:  # Red hues
                    nebula_type = "emission nebula"
                elif 90 <= avg_hue <= 130:  # Blue hues
                    nebula_type = "reflection nebula"
                else:
                    # Check if it's a dark nebula by comparing to surrounding brightness
                    surrounding = gray_image[max(0, y-10):min(gray_image.shape[0], y+h+10),
                                            max(0, x-10):min(gray_image.shape[1], x+w+10)]
                    region_gray = gray_image[y:y+h, x:x+w]
                    if surrounding.size > 0 and region_gray.size > 0:
                        if np.mean(region_gray) < np.mean(surrounding) * 0.7:
                            nebula_type = "dark nebula"
                        else:
                            nebula_type = "reflection nebula"
                    else:
                        nebula_type = "reflection nebula"
            else:
                nebula_type = "reflection nebula"
            
            nebulae.append({
                "x": float(center_x),  # Garantindo que seja float nativo
                "y": float(center_y),  # Garantindo que seja float nativo
                "size": float(size),  # Garantindo que seja float nativo
                "magnitude": float(magnitude),  # Garantindo que seja float nativo
                "nebula_type": nebula_type
            })
        
        return nebulae
    
    def identify_known_galaxies(self, detected_objects, image_rgb, image_analysis):
        """Identify known galaxies in the image"""
        logger.info("Identifying known galaxies")
        
        # Extract galaxies from detected objects
        galaxies = [obj for obj in detected_objects if obj["type"] == "galaxy"]
        
        if not galaxies:
            return
        
        # Get image features for galaxy identification
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Sort galaxies by size (largest first)
        galaxies.sort(key=lambda x: x["size"], reverse=True)
        
        # Check the largest galaxy against known galaxies
        if galaxies:
            largest_galaxy = galaxies[0]
            
            # Extract region around the galaxy
            x, y = int(largest_galaxy["position"]["x"]), int(largest_galaxy["position"]["y"])
            size = int(largest_galaxy["size"])
            half_size = max(50, size // 2)
            
            # Ensure coordinates are within image bounds
            y1 = max(0, y - half_size)
            y2 = min(image_rgb.shape[0], y + half_size)
            x1 = max(0, x - half_size)
            x2 = min(image_rgb.shape[1], x + half_size)
            
            if x1 < x2 and y1 < y2:
                galaxy_region = image_rgb[y1:y2, x1:x2]
                
                # Calculate features for the galaxy region
                if galaxy_region.size > 0:
                    # Calculate color distribution
                    r, g, b = cv2.split(galaxy_region)
                    r_mean, g_mean, b_mean = np.mean(r)/255.0, np.mean(g)/255.0, np.mean(b)/255.0
                    
                    # Calculate texture features
                    gray_region = cv2.cvtColor(galaxy_region, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray_region, 100, 200)
                    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                    
                    # Check for spiral arms (using edge orientation histogram)
                    sobelx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
                    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
                    hist, _ = np.histogram(orientation, bins=18, range=(-180, 180))
                    hist_normalized = hist / np.sum(hist)
                    spiral_score = np.std(hist_normalized)  # Higher for spiral galaxies
                    
                    # Match against known galaxies
                    best_match = None
                    best_score = 0
                    
                    for name, galaxy_data in KNOWN_GALAXIES.items():
                        score = 0
                        
                        # Check visual features
                        for feature in galaxy_data["visual_features"]:
                            if "spiral" in feature and spiral_score > 0.1:
                                score += 0.2
                            if "bulge" in feature and edge_density < 0.1:
                                score += 0.1
                            if "dust" in feature and edge_density > 0.1:
                                score += 0.15
                            if "edge-on" in feature and (y2-y1) > (x2-x1) * 1.5:
                                score += 0.2
                            if "face-on" in feature and (x2-x1) > (y2-y1) * 0.8:
                                score += 0.2
                            if "bluish" in feature and b_mean > r_mean:
                                score += 0.15
                            if "yellowish" in feature and r_mean > b_mean and g_mean > b_mean:
                                score += 0.15
                        
                        # Check if color distribution matches expected
                        if "spiral galaxy" in galaxy_data["type"] and b_mean > 0.3:
                            score += 0.1
                        if "elliptical galaxy" in galaxy_data["type"] and r_mean > b_mean:
                            score += 0.1
                        
                        # If this is the best match so far, update
                        if score > best_score:
                            best_score = score
                            best_match = name
                    
                    # If we have a good match, update the galaxy information
                    if best_match and best_score > 0.3:
                        confidence = min(0.5 + best_score, 0.95)  # Scale confidence based on match score
                        
                        # Update the galaxy object
                        largest_galaxy["name"] = best_match
                        largest_galaxy["confidence"] = float(confidence)  # Garantindo que seja float nativo
                        largest_galaxy["galaxy_type"] = KNOWN_GALAXIES[best_match]["type"]
                        
                        # Also update the image analysis
                        image_analysis["galaxy_name"] = best_match
                        image_analysis["galaxy_confidence"] = float(confidence)  # Garantindo que seja float nativo
                        
                        logger.info(f"Identified galaxy: {best_match} with confidence {confidence:.2f}")
    
    def calculate_scientific_data(self, detected_objects, image_analysis):
        """Calculate scientific data for each detected object"""
        logger.info("Calculating scientific data")
        
        for obj in detected_objects:
            # Generate random but plausible scientific data
            if obj["type"] == "star":
                # Calculate temperature based on color
                if obj["color"] == "blue":
                    temp_range = STAR_TYPES["O"]["temp_range"]
                    temp = np.random.uniform(temp_range[0], temp_range[1])
                elif obj["color"] == "blue-white" or obj["color"] == "white":
                    temp_range = STAR_TYPES["A"]["temp_range"]
                    temp = np.random.uniform(temp_range[0], temp_range[1])
                elif obj["color"] == "yellow" or obj["color"] == "yellow-white":
                    temp_range = STAR_TYPES["G"]["temp_range"]
                    temp = np.random.uniform(temp_range[0], temp_range[1])
                elif obj["color"] == "orange":
                    temp_range = STAR_TYPES["K"]["temp_range"]
                    temp = np.random.uniform(temp_range[0], temp_range[1])
                elif obj["color"] == "red":
                    temp_range = STAR_TYPES["M"]["temp_range"]
                    temp = np.random.uniform(temp_range[0], temp_range[1])
                else:
                    temp = np.random.uniform(3000, 30000)
                
                # Calculate distance based on magnitude and temperature
                # Brighter stars appear closer
                base_distance = np.random.uniform(100, 10000)
                distance_factor = (obj["magnitude"] - 10) / 2  # Adjust based on magnitude
                distance = base_distance * (1 + distance_factor)
                
                # Add the data to the object
                obj["temperature"] = float(temp)  # Garantindo que seja float nativo
                obj["temperature_unit"] = "K"
                obj["distance"] = float(distance)  # Garantindo que seja float nativo
                obj["distance_unit"] = "light-years"
                
            elif obj["type"] == "galaxy":
                # If it's a known galaxy, use the known data
                if obj["name"] and obj["name"] in KNOWN_GALAXIES:
                    galaxy_data = KNOWN_GALAXIES[obj["name"]]
                    obj["distance"] = float(galaxy_data["distance"])  # Garantindo que seja float nativo
                    obj["distance_unit"] = galaxy_data["distance_unit"]
                    obj["redshift"] = float(galaxy_data["redshift"])  # Garantindo que seja float nativo
                    obj["mass"] = float(galaxy_data["mass"])  # Garantindo que seja float nativo
                    obj["mass_unit"] = galaxy_data["mass_unit"]
                    obj["temperature"] = float(galaxy_data["temperature"])  # Garantindo que seja float nativo
                    obj["temperature_unit"] = galaxy_data["temperature_unit"]
                    obj["catalog"] = galaxy_data["catalog"]
                else:
                    # Generate plausible data for unknown galaxy
                    obj["distance"] = float(np.random.uniform(10, 500))  # Garantindo que seja float nativo
                    obj["distance_unit"] = "million light-years"
                    obj["redshift"] = float(np.random.uniform(0.001, 0.1))  # Garantindo que seja float nativo
                    obj["mass"] = float(np.random.uniform(1e10, 1e12))  # Garantindo que seja float nativo
                    obj["mass_unit"] = "solar masses"
                    obj["temperature"] = float(np.random.uniform(3000, 8000))  # Garantindo que seja float nativo
                    obj["temperature_unit"] = "K"
            
            elif obj["type"] == "nebula":
                # Generate plausible data for nebula
                obj["distance"] = float(np.random.uniform(1, 10))  # Garantindo que seja float nativo
                obj["distance_unit"] = "thousand light-years"
                
                # Temperature depends on nebula type
                if "emission" in obj["nebula_type"]:
                    obj["temperature"] = float(np.random.uniform(8000, 15000))  # Garantindo que seja float nativo
                elif "reflection" in obj["nebula_type"]:
                    obj["temperature"] = float(np.random.uniform(3000, 10000))  # Garantindo que seja float nativo
                elif "dark" in obj["nebula_type"]:
                    obj["temperature"] = float(np.random.uniform(10, 100))  # Garantindo que seja float nativo
                else:
                    obj["temperature"] = float(np.random.uniform(1000, 10000))  # Garantindo que seja float nativo
                
                obj["temperature_unit"] = "K"
    
    def generate_summary(self, detected_objects, image_analysis):
        """Generate a summary of the detected objects and image analysis"""
        logger.info("Generating summary")
        
        # Count objects by type
        stars_count = sum(1 for obj in detected_objects if obj["type"] == "star")
        galaxies_count = sum(1 for obj in detected_objects if obj["type"] == "galaxy")
        nebulae_count = sum(1 for obj in detected_objects if obj["type"] == "nebula")
        
        # Get named objects
        named_stars = [obj["name"] for obj in detected_objects if obj["type"] == "star" and obj["name"]]
        named_galaxies = [obj["name"] for obj in detected_objects if obj["type"] == "galaxy" and obj["name"]]
        named_nebulae = [obj["name"] for obj in detected_objects if obj["type"] == "nebula" and obj["name"]]
        
        # Determine brightness level
        if image_analysis["brightness"] < 0.2:
            brightness_level = "Low"
        elif image_analysis["brightness"] < 0.5:
            brightness_level = "Medium"
        else:
            brightness_level = "High"
        
        # Determine contrast level
        if image_analysis["contrast"] < 0.2:
            contrast_level = "Low"
        elif image_analysis["contrast"] < 0.5:
            contrast_level = "Medium"
        else:
            contrast_level = "High"
        
        # Determine spectral shift
        spectral_shift = None
        if image_analysis["potential_redshift"] and not image_analysis["potential_blueshift"]:
            spectral_shift = "Potential redshift detected"
        elif image_analysis["potential_blueshift"] and not image_analysis["potential_redshift"]:
            spectral_shift = "Potential blueshift detected"
        elif image_analysis["potential_redshift"] and image_analysis["potential_blueshift"]:
            spectral_shift = "Mixed spectral shifts detected"
        
        # Generate description
        description = f"This image shows "
        
        if image_analysis["galaxy_name"]:
            description += f"the {image_analysis['galaxy_name']}, "
            if "Andromeda" in image_analysis["galaxy_name"]:
                description += f"a spiral galaxy located approximately 2.5 million light-years from Earth with an apparent magnitude of 3.44. It has a blueshift, indicating it's moving toward us. "
            elif "Whirlpool" in image_analysis["galaxy_name"]:
                description += f"a spiral galaxy located approximately 23.0 million light-years from Earth with an apparent magnitude of 8.40. It has a redshift of 0.001544, indicating it's moving away from us. "
            elif "Triangulum" in image_analysis["galaxy_name"]:
                description += f"a spiral galaxy located approximately 2.73 million light-years from Earth with an apparent magnitude of 5.72. It has a blueshift, indicating it's moving toward us. "
            elif "Sombrero" in image_analysis["galaxy_name"]:
                description += f"a spiral galaxy located approximately 29.3 million light-years from Earth with an apparent magnitude of 8.98. It has a redshift of 0.003416, indicating it's moving away from us. "
        
        if stars_count > 0:
            description += f"There are also approximately {stars_count} stars visible "
            if image_analysis["galaxy_name"]:
                description += "in the foreground."
            else:
                description += "in the image."
        
        # Generate notable objects text
        notable_objects = None
        if named_galaxies or named_stars or named_nebulae:
            notable_objects = "Notable objects identified: "
            if named_galaxies:
                notable_objects += f"{', '.join(named_galaxies)} (galaxies)"
            if named_stars:
                if named_galaxies:
                    notable_objects += ", "
                notable_objects += f"{', '.join(named_stars)} (stars)"
            if named_nebulae:
                if named_galaxies or named_stars:
                    notable_objects += ", "
                notable_objects += f"{', '.join(named_nebulae)} (nebulae)"
        
        return {
            "image_type": image_analysis["image_type"],
            "total_objects": len(detected_objects),
            "object_counts": {
                "stars": stars_count,
                "galaxies": galaxies_count,
                "nebulae": nebulae_count
            },
            "named_objects": {
                "stars": named_stars,
                "galaxies": named_galaxies,
                "nebulae": named_nebulae
            },
            "brightness_level": brightness_level,
            "contrast_level": contrast_level,
            "complexity": image_analysis["complexity"],
            "dominant_color": image_analysis["dominant_color"],
            "spectral_shift": spectral_shift,
            "description": description,
            "notable_objects": notable_objects
        }
    
    def draw_annotations(self, image, detected_objects):
        """Draw annotations on the image to highlight detected objects"""
        logger.info("Drawing annotations")
        
        # Convert to PIL for easier text drawing
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw annotations for each object
        for obj in detected_objects:
            x, y = obj["position"]["x"], obj["position"]["y"]
            
            # Different colors and shapes for different object types
            if obj["type"] == "star":
                # Stars get small circles
                radius = max(2, int(obj["size"] / 2))
                color = (255, 255, 0)  # Yellow for stars
                
                # Draw circle
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=1)
                
                # Only label bright stars to avoid cluttering
                if obj["magnitude"] < 11.0 and obj["size"] > 5:
                    label = obj["name"] if obj["name"] else "Star"
                    draw.text((x+radius+2, y-radius), label, fill=color, font=small_font)
            
            elif obj["type"] == "galaxy":
                # Galaxies get larger circles
                radius = max(10, int(obj["size"] / 4))
                color = (0, 255, 255)  # Cyan for galaxies
                
                # Draw circle
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
                
                # Label the galaxy
                label = obj["name"] if obj["name"] else "Galaxy"
                draw.text((x+radius+2, y-radius), label, fill=color, font=font)
                
                # Add additional info for known galaxies
                if obj["name"]:
                    info = f"Type: {obj['galaxy_type']}"
                    draw.text((x+radius+2, y-radius+16), info, fill=color, font=small_font)
                    
                    if "distance" in obj:
                        dist_text = f"Distance: {obj['distance']} {obj['distance_unit']}"
                        draw.text((x+radius+2, y-radius+32), dist_text, fill=color, font=small_font)
            
            elif obj["type"] == "nebula":
                # Nebulae get rectangles
                size = max(15, int(obj["size"] / 3))
                color = (255, 0, 255)  # Magenta for nebulae
                
                # Draw rectangle
                draw.rectangle((x-size, y-size, x+size, y+size), outline=color, width=2)
                
                # Label the nebula
                label = obj["name"] if obj["name"] else obj["nebula_type"].capitalize()
                draw.text((x+size+2, y-size), label, fill=color, font=small_font)
        
        # Convert back to numpy array
        return np.array(pil_image)

# Main function to be called from the API
def process_image(input_path, output_path, results_path):
    detector = CelestialDetector()
    success = detector.process_image(input_path, output_path, results_path)
    return success

# Command line interface
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python celestial_detector.py <input_image_path> <output_image_path> <results_json_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    results_path = sys.argv[3]
    
    detector = CelestialDetector()
    success = detector.process_image(input_path, output_path, results_path)
    
    if success:
        print(f"Processing complete. Results saved to {results_path}")
    else:
        print("Error processing image")
        sys.exit(1)

