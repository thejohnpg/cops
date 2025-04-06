#!/usr/bin/env python3
"""
Celestial Body Detection System - Simplified Version
This script uses basic computer vision techniques to analyze astronomical images.
"""

import os
import sys
import json
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import math
import time
import logging
import re
import requests
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for astronomical catalogs and APIs
SIMBAD_BASE_URL = "http://simbad.u-strasbg.fr/simbad/sim-script"

# Astronomical object types
OBJECT_TYPES = {
    'galaxy': ['galaxy', 'spiral galaxy', 'elliptical galaxy', 'lenticular galaxy', 'irregular galaxy'],
    'nebula': ['nebula', 'emission nebula', 'reflection nebula', 'dark nebula', 'planetary nebula'],
    'star': ['star', 'binary star', 'variable star', 'neutron star', 'white dwarf', 'red giant'],
    'star_cluster': ['open cluster', 'globular cluster', 'star cluster']
}

# Known astronomical objects with their characteristics
KNOWN_OBJECTS = {
    'Andromeda Galaxy': {
        'type': 'galaxy',
        'subtype': 'spiral galaxy',
        'catalog': 'M31, NGC 224',
        'distance': 2.537,  # million light-years
        'magnitude': 3.44,
        'redshift': -0.001001,  # blueshift
        'size': 3.167,  # degrees (angular diameter)
        'mass': 1.5e12,  # solar masses
        'description': "The Andromeda Galaxy is the nearest major galaxy to the Milky Way."
    },
    'Triangulum Galaxy': {
        'type': 'galaxy',
        'subtype': 'spiral galaxy',
        'catalog': 'M33, NGC 598',
        'distance': 2.73,  # million light-years
        'magnitude': 5.72,
        'redshift': -0.000607,  # blueshift
        'size': 1.0,  # degrees (angular diameter)
        'mass': 5.0e10,  # solar masses
        'description': "The Triangulum Galaxy is the third-largest member of the Local Group."
    },
    'Orion Nebula': {
        'type': 'nebula',
        'subtype': 'emission nebula',
        'catalog': 'M42, NGC 1976',
        'distance': 1.344,  # thousand light-years
        'magnitude': 4.0,
        'size': 1.0,  # degrees (angular diameter)
        'description': "The Orion Nebula is a diffuse nebula situated in the Milky Way."
    },
    'Pleiades': {
        'type': 'star_cluster',
        'subtype': 'open cluster',
        'catalog': 'M45',
        'distance': 0.444,  # thousand light-years
        'magnitude': 1.6,
        'size': 2.0,  # degrees (angular diameter)
        'description': "The Pleiades is an open star cluster in the constellation Taurus."
    }
}

class CelestialObjectDetector:
    def __init__(self):
        """Initialize the detector with necessary resources."""
        logger.info("Initializing Celestial Object Detector")
        
        # Load galaxy templates for pattern matching
        self.galaxy_templates = self._load_galaxy_templates()
        
    def _load_galaxy_templates(self):
        """Load template features for known galaxies."""
        return {
            "Andromeda": {
                "ellipticity": 0.7,  # ratio of minor to major axis
                "brightness_profile": "exponential",  # brightness decreases exponentially from center
                "size_ratio": 0.3,  # ratio of galaxy size to image size
                "features": ["bright_core", "spiral_arms", "dust_lanes"]
            },
            "Triangulum": {
                "ellipticity": 0.9,  # more circular
                "brightness_profile": "exponential",
                "size_ratio": 0.25,
                "features": ["spiral_arms", "star_forming_regions"]
            },
            "Whirlpool": {
                "ellipticity": 0.85,
                "brightness_profile": "exponential",
                "size_ratio": 0.2,
                "features": ["spiral_arms", "companion_galaxy"]
            },
            "Sombrero": {
                "ellipticity": 0.3,  # highly elliptical (edge-on)
                "brightness_profile": "de_vaucouleurs",
                "size_ratio": 0.15,
                "features": ["dust_lane", "bright_core", "extended_halo"]
            }
        }
    
    def query_astronomical_database(self, object_name):
        """Query astronomical databases for information about a specific object."""
        try:
            # First check our local database
            if object_name in KNOWN_OBJECTS:
                return KNOWN_OBJECTS[object_name]
            
            # Try to query SIMBAD
            try:
                encoded_name = quote_plus(object_name)
                script = f"format object form1 \"Object:%IDLIST(1)[%*,]\\nType:%OTYPE\\nCoordinates:%COORD\\nMagnitude:%FLUXLIST(V)\\nRedshift:%REDSHIFT\\nDistance:%DIST\\n\"\nquery id {encoded_name}"
                params = {
                    'script': script
                }
                
                response = requests.post(SIMBAD_BASE_URL, data=params, timeout=5)
                
                if response.status_code == 200 and "No astronomical object found" not in response.text:
                    result = {}
                    lines = response.text.strip().split('\n')
                    
                    for line in lines:
                        if line.startswith('Object:'):
                            result['catalog'] = line.replace('Object:', '').strip()
                        elif line.startswith('Type:'):
                            obj_type = line.replace('Type:', '').strip().lower()
                            for key, values in OBJECT_TYPES.items():
                                if any(val in obj_type for val in values):
                                    result['type'] = key
                                    result['subtype'] = obj_type
                                    break
                            else:
                                result['type'] = 'unknown'
                                result['subtype'] = obj_type
                        elif line.startswith('Magnitude:'):
                            mag_str = line.replace('Magnitude:', '').strip()
                            if mag_str and mag_str != '~':
                                try:
                                    result['magnitude'] = float(mag_str)
                                except ValueError:
                                    pass
                        elif line.startswith('Redshift:'):
                            z_str = line.replace('Redshift:', '').strip()
                            if z_str and z_str != '~':
                                try:
                                    result['redshift'] = float(z_str)
                                except ValueError:
                                    pass
                    
                    if result:
                        return result
            except Exception as e:
                logger.warning(f"SIMBAD query failed: {e}")
            
            # If no result from SIMBAD, return a basic template
            return {
                'type': 'unknown',
                'subtype': 'unknown',
                'catalog': object_name,
                'description': f"Information about {object_name} could not be found in astronomical databases."
            }
            
        except Exception as e:
            logger.error(f"Error querying astronomical database: {e}")
            return {
                'type': 'unknown',
                'subtype': 'unknown',
                'catalog': object_name,
                'description': f"Error retrieving information about {object_name}."
            }
    
    def preprocess_image(self, image):
        """Preprocess the image for analysis."""
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # RGB
            image_rgb = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Create normalized versions for different analyses
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        image_normalized = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return {
            'original': image,
            'rgb': image_rgb,
            'gray': image_gray,
            'normalized': image_normalized
        }
    
    def analyze_image_content(self, image_dict):
        """Analyze the image to determine its general content."""
        try:
            # Use traditional computer vision methods
            image_type = self.analyze_image_traditional(image_dict)
            confidence = 0.7  # Default confidence
            
            # Calculate basic image statistics
            gray = image_dict['gray']
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Detect edges to estimate complexity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.1:
                complexity = "high"
            elif edge_density > 0.05:
                complexity = "medium"
            else:
                complexity = "low"
            
            # Estimate color distribution
            rgb = image_dict['rgb']
            r_mean = np.mean(rgb[:,:,0])
            g_mean = np.mean(rgb[:,:,1])
            b_mean = np.mean(rgb[:,:,2])
            
            # Determine dominant color
            if r_mean > g_mean and r_mean > b_mean:
                dominant_color = "red"
            elif g_mean > r_mean and g_mean > b_mean:
                dominant_color = "green"
            elif b_mean > r_mean and b_mean > g_mean:
                dominant_color = "blue"
            else:
                dominant_color = "balanced"
                
            # Calculate color ratios for potential redshift/blueshift estimation
            if r_mean > 0 and b_mean > 0:
                rb_ratio = r_mean / b_mean
                potential_redshift = rb_ratio > 1.2
                potential_blueshift = rb_ratio < 0.8
            else:
                potential_redshift = False
                potential_blueshift = False
            
            return {
                "image_type": image_type,
                "confidence": float(confidence),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "complexity": complexity,
                "edge_density": float(edge_density),
                "dominant_color": dominant_color,
                "potential_redshift": bool(potential_redshift),
                "potential_blueshift": bool(potential_blueshift),
                "color_distribution": {
                    "red": float(r_mean),
                    "green": float(g_mean),
                    "blue": float(b_mean)
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_image_content: {e}")
            # Return a basic analysis if the approach fails
            return {
                "image_type": "unknown",
                "confidence": 0.5,
                "brightness": 100.0,
                "contrast": 50.0,
                "complexity": "medium",
                "edge_density": 0.05,
                "dominant_color": "balanced",
                "potential_redshift": False,
                "potential_blueshift": False,
                "color_distribution": {
                    "red": 100.0,
                    "green": 100.0,
                    "blue": 100.0
                }
            }
    
    def analyze_image_traditional(self, image_dict):
        """Analyze the image using traditional computer vision methods."""
        try:
            # Get the grayscale image
            gray = image_dict['gray']
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding to find bright spots (potential stars)
            _, thresh = cv2.threshold(blurred, np.mean(blurred) * 1.5, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count small bright spots (potential stars)
            star_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Small areas are likely stars
                    star_count += 1
            
            # Check for large central bright region (typical of galaxy cores)
            has_central_object = False
            height, width = gray.shape
            center_region = gray[height//4:3*height//4, width//4:3*width//4]
            center_brightness = np.mean(center_region)
            overall_brightness = np.mean(gray)
            
            if center_brightness > overall_brightness * 1.5:
                has_central_object = True
            
            # Determine image type based on characteristics
            if has_central_object and star_count < 50:
                return "galaxy"
            elif star_count > 100:
                return "star field"
            elif has_central_object and star_count > 50:
                return "nebula or galaxy cluster"
            else:
                return "deep space"
        except Exception as e:
            logger.error(f"Error in analyze_image_traditional: {e}")
            return "unknown"
    
    def detect_stars(self, image_dict):
        """Detect stars in the image."""
        try:
            # Get the grayscale image
            gray = image_dict['normalized']
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding to find bright spots
            _, thresh = cv2.threshold(blurred, np.mean(blurred) * 1.5, 255, cv2.THRESH_BINARY)
            
            # Find contours (potential stars)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape to identify stars
            stars = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Small areas are likely stars
                    # Get the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate brightness (mean pixel value in the original image)
                        mask = np.zeros_like(gray)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        brightness = np.mean(gray[mask == 255]) if np.sum(mask == 255) > 0 else 0
                        
                        # Calculate size based on contour area
                        size = math.sqrt(area)
                        
                        # Estimate magnitude based on brightness
                        # This is a simplified approach - real magnitude calculation would be more complex
                        magnitude = 20 - 5 * math.log10(brightness + 1)
                        
                        # Add the star with its position
                        stars.append({
                            "type": "star",
                            "position": {"x": cx, "y": cy},
                            "size": float(size),
                            "brightness": float(brightness),
                            "magnitude": float(magnitude),
                            "confidence": 0.8
                        })
            
            return stars
        except Exception as e:
            logger.error(f"Error in detect_stars: {e}")
            return []
    
    def detect_galaxy_type(self, image):
        """Detect if the image contains a known galaxy and identify its type."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Use adaptive thresholding to find bright regions
        _, thresh = cv2.threshold(blurred, np.mean(blurred) * 1.5, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for large central bright region (typical of galaxy cores)
        has_central_object = False
        ellipticity = 0
        size_ratio = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.01):  # At least 1% of image area
                # Get the center of the contour
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Check if it's near the center of the image
                    center_x, center_y = width // 2, height // 2
                    distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    if distance_from_center < (width + height) / 8:  # Within central region
                        has_central_object = True
                        
                        # Fit an ellipse to the contour if possible
                        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                            ellipse = cv2.fitEllipse(contour)
                            (_, _), (major_ = cv2.fitEllipse(contour)
                            (_, _), (major_axis, minor_axis), _ = ellipse
                            
                            # Calculate ellipticity (ratio of minor to major axis)
                            if major_axis > 0:
                                ellipticity = minor_axis / major_axis
                            
                            # Calculate size ratio (major axis relative to image size)
                            size_ratio = major_axis / min(width, height)
                        
                        break
        
        # Check brightness profile (exponential decay is typical for spiral galaxies)
        brightness_profile = "unknown"
        if has_central_object:
            center_x, center_y = width // 2, height // 2
            max_radius = min(width, height) // 2
            radii = np.arange(1, max_radius, max_radius // 20)
            brightness_values = []
            
            for r in radii:
                y, x = np.ogrid[:height, :width]
                mask = ((x - center_x)**2 + (y - center_y)**2 <= r**2) & \
                       ((x - center_x)**2 + (y - center_y)**2 > (r-1)**2)
                if np.any(mask):
                    brightness_values.append(np.mean(gray[mask]))
            
            # Check if brightness decreases exponentially from center
            if len(brightness_values) > 3:
                # Check if values are generally decreasing
                is_decreasing = all(brightness_values[i] >= brightness_values[i+1] 
                                   for i in range(len(brightness_values)-1))
                
                # Check rate of decrease
                if is_decreasing:
                    # Calculate ratio of consecutive values
                    ratios = [brightness_values[i]/brightness_values[i+1] if brightness_values[i+1] > 0 else 1 
                             for i in range(len(brightness_values)-1)]
                    
                    # If ratios are relatively constant, it's exponential decay
                    if np.std(ratios) < 0.3 * np.mean(ratios):
                        brightness_profile = "exponential"
                    else:
                        brightness_profile = "de_vaucouleurs"  # Steeper in center, flatter in outskirts
        
        # Detect features
        features = []
        
        # Check for spiral arms (look for curved structures)
        # This is a simplified approach - real detection would be more complex
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Count curved vs straight lines
            curved_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check points along the line
                points = np.linspace(0, 1, 10)
                line_points = np.array([(int(x1 + p*(x2-x1)), int(y1 + p*(y2-y1))) for p in points])
                
                # Check if points deviate from a straight line
                if len(line_points) >= 3:
                    # Fit a line to the points
                    vx, vy, x0, y0 = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate distances from points to the fitted line
                    distances = []
                    for point in line_points:
                        # Distance from point to line
                        dist = abs((vy * (point[0] - x0) - vx * (point[1] - y0)) / 
                                  np.sqrt(vx*vx + vy*vy))
                        distances.append(dist)
                    
                    # If max distance is significant, it's curved
                    if max(distances) > 5:  # Threshold for curvature
                        curved_count += 1
            
            if curved_count > len(lines) * 0.3:  # If more than 30% of lines are curved
                features.append("spiral_arms")
        
        # Check for bright core
        center_brightness = np.mean(gray[height//2-10:height//2+10, width//2-10:width//2+10])
        overall_brightness = np.mean(gray)
        
        if center_brightness > overall_brightness * 2:
            features.append("bright_core")
        
        # Check for dust lanes (dark streaks across bright regions)
        if has_central_object:
            # Apply stronger threshold to find very bright regions
            _, bright_regions = cv2.threshold(blurred, np.mean(blurred) * 2, 255, cv2.THRESH_BINARY)
            
            # Look for dark regions within bright areas
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(bright_regions, kernel, iterations=2)
            potential_dust = cv2.bitwise_xor(dilated, bright_regions)
            
            # Count potential dust pixels
            dust_pixel_count = np.sum(potential_dust > 0)
            if dust_pixel_count > (width * height * 0.01):
                features.append("dust_lanes")
        
        # Match against known galaxy templates
        best_match = None
        best_score = 0
        
        for galaxy_name, template in self.galaxy_templates.items():
            score = 0
            
            # Compare ellipticity
            ellipticity_diff = abs(template["ellipticity"] - ellipticity)
            if ellipticity_diff < 0.2:
                score += (1 - ellipticity_diff * 5) * 0.3  # 30% weight
            
            # Compare brightness profile
            if template["brightness_profile"] == brightness_profile:
                score += 0.3  # 30% weight
            
            # Compare size ratio
            size_diff = abs(template["size_ratio"] - size_ratio)
            if size_diff < 0.15:
                score += (1 - size_diff * 6.67) * 0.2  # 20% weight
            
            # Compare features
            feature_match_count = sum(1 for f in features if f in template["features"])
            feature_score = feature_match_count / max(len(template["features"]), 1)
            score += feature_score * 0.2  # 20% weight
            
            if score > best_score:
                best_score = score
                best_match = galaxy_name
        
        # Special case for Andromeda - it's very distinctive
        if "bright_core" in features and "spiral_arms" in features and ellipticity < 0.8 and ellipticity > 0.5:
            andromeda_likelihood = 0.7
            if "dust_lanes" in features:
                andromeda_likelihood += 0.2
            
            if best_match == "Andromeda" or andromeda_likelihood > 0.8:
                return "Andromeda", best_score
        
        if best_match and best_score > 0.6:
            return best_match, best_score
        elif has_central_object and "bright_core" in features:
            return "unknown_galaxy", 0.5
        else:
            return None, 0
    
    def detect_galaxies(self, image_dict, image_analysis):
        """Detect galaxies in the image."""
        try:
            # If the image is not likely to contain a galaxy, return empty
            if image_analysis["image_type"] != "galaxy" and image_analysis["confidence"] > 0.7:
                return []
            
            # Get the original image
            original = image_dict['original']
            height, width = original.shape[:2]
            
            # Detect galaxy type
            galaxy_name, confidence = self.detect_galaxy_type(original)
            
            # If we detected a specific galaxy
            if galaxy_name and galaxy_name != "unknown_galaxy":
                # Query database for galaxy info
                if galaxy_name == "Andromeda":
                    galaxy_info = self.query_astronomical_database("Andromeda Galaxy")
                elif galaxy_name == "Triangulum":
                    galaxy_info = self.query_astronomical_database("Triangulum Galaxy")
                else:
                    galaxy_info = self.query_astronomical_database(f"{galaxy_name} Galaxy")
                
                # Create galaxy object
                galaxy = {
                    "type": "galaxy",
                    "name": f"{galaxy_name} Galaxy",
                    "galaxy_type": galaxy_info.get("subtype", "spiral galaxy"),
                    "position": {"x": width // 2, "y": height // 2},
                    "size": min(width, height) / 3,
                    "catalog": galaxy_info.get("catalog", ""),
                    "distance": galaxy_info.get("distance", 0),
                    "distance_unit": "million light-years",
                    "redshift": galaxy_info.get("redshift", 0),
                    "mass": galaxy_info.get("mass", 0),
                    "mass_unit": "solar masses",
                    "description": galaxy_info.get("description", ""),
                    "confidence": float(confidence)
                }
                
                return [galaxy]
            
            # Otherwise, detect galaxies using traditional methods
            gray = image_dict['normalized']
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # Use adaptive thresholding to find bright regions
            _, thresh = cv2.threshold(blurred, np.mean(blurred) * 1.2, 255, cv2.THRESH_BINARY)
            
            # Find contours (potential galaxies)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            galaxies = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (width * height * 0.01):  # At least 1% of image area
                    # Get the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if it's near the center of the image
                        center_x, center_y = width // 2, height // 2
                        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        
                        if distance_from_center < (width + height) / 8:  # Within central region
                            # Fit an ellipse to the contour if possible
                            if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                                ellipse = cv2.fitEllipse(contour)
                                (_, _), (major_axis, minor_axis), angle = ellipse
                                
                                # Calculate ellipticity (ratio of minor to major axis)
                                ellipticity = minor_axis / major_axis if major_axis > 0 else 0
                                
                                # Determine galaxy type based on shape
                                if ellipticity > 0.8:
                                    galaxy_type = "elliptical"
                                elif ellipticity > 0.3:
                                    galaxy_type = "spiral"
                                else:
                                    galaxy_type = "edge-on spiral"
                                
                                # Calculate size based on major axis
                                size = major_axis
                                
                                # Estimate magnitude based on brightness
                                brightness = np.mean(gray[cy-10:cy+10, cx-10:cx+10])
                                magnitude = 15 - 2.5 * math.log10(brightness + 1)
                                
                                # Add the galaxy with its position
                                galaxies.append({
                                    "type": "galaxy",
                                    "galaxy_type": galaxy_type,
                                    "position": {"x": cx, "y": cy},
                                    "size": float(size),
                                    "ellipticity": float(ellipticity),
                                    "angle": float(angle),
                                    "magnitude": float(magnitude),
                                    "confidence": 0.7
                                })
            
            return galaxies
        except Exception as e:
            logger.error(f"Error in detect_galaxies: {e}")
            return []
    
    def detect_nebulae(self, image_dict, image_analysis):
        """Detect nebulae in the image."""
        try:
            # Get the grayscale image
            gray = image_dict['normalized']
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # Use adaptive thresholding to find bright regions
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 51, 10)
            
            # Find contours (potential nebulae)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            nebulae = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Larger areas could be nebulae
                    # Get the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate perimeter and circularity
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        
                        # Determine nebula type based on shape and color
                        rgb = image_dict['rgb']
                        mask = np.zeros_like(gray)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        
                        # Get color in the nebula region
                        r_mean = np.mean(rgb[:,:,0][mask == 255]) if np.sum(mask == 255) > 0 else 0
                        g_mean = np.mean(rgb[:,:,1][mask == 255]) if np.sum(mask == 255) > 0 else 0
                        b_mean = np.mean(rgb[:,:,2][mask == 255]) if np.sum(mask == 255) > 0 else 0
                        
                        if r_mean > g_mean and r_mean > b_mean:
                            nebula_type = "emission"  # Red often indicates emission nebulae
                        elif b_mean > r_mean and b_mean > g_mean:
                            nebula_type = "reflection"  # Blue often indicates reflection nebulae
                        elif circularity > 0.7:
                            nebula_type = "planetary"  # More circular objects could be planetary nebulae
                        else:
                            nebula_type = "dark"  # Default to dark nebula
                        
                        # Calculate size based on contour area
                        size = math.sqrt(area)
                        
                        # Estimate magnitude
                        brightness = np.mean(gray[mask == 255]) if np.sum(mask == 255) > 0 else 0
                        magnitude = 15 - 2.5 * math.log10(brightness + 1)
                        
                        # Add the nebula with its position
                        nebulae.append({
                            "type": "nebula",
                            "nebula_type": nebula_type,
                            "position": {"x": cx, "y": cy},
                            "size": float(size),
                            "circularity": float(circularity),
                            "magnitude": float(magnitude),
                            "confidence": 0.7
                        })
            
            return nebulae
        except Exception as e:
            logger.error(f"Error in detect_nebulae: {e}")
            return []
    
    def create_visualization(self, image, objects):
        """Create a visualization of the detected objects with scientific labels."""
        try:
            # Convert OpenCV image to PIL for easier drawing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                small_font = ImageFont.truetype("arial.ttf", 10)
            except IOError:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw the detected objects
            for obj in objects:
                # Get position
                if "position" in obj:
                    x, y = obj["position"]["x"], obj["position"]["y"]
                else:
                    # Skip objects without position
                    continue
                
                size = int(obj["size"]) if "size" in obj else 10
                
                # Choose color and style based on object type
                if obj["type"] == "star":
                    color = (255, 255, 0)  # Yellow for stars
                    draw.ellipse((x-size-2, y-size-2, x+size+2, y+size+2), outline=(255, 255, 0))
                    # Add crosshairs for stars
                    draw.line((x-10, y, x+10, y), fill=(255, 255, 0))
                    draw.line((x, y-10, x, y+10), fill=(255, 255, 0))
                elif obj["type"] == "galaxy":
                    color = (138, 43, 226)  # Purple for galaxies
                    # Draw ellipse for galaxies
                    if "galaxy_type" in obj and obj["galaxy_type"] == "spiral":
                        # Draw spiral pattern for spiral galaxies
                        for i in range(0, 360, 30):
                            angle = math.radians(i)
                            end_x = x + int(size * 1.5 * math.cos(angle))
                            end_y = y + int(size * 1.5 * math.sin(angle))
                            draw.line((x, y, end_x, end_y), fill=(138, 43, 226))
                        
                        # Draw central bulge
                        draw.ellipse((x-size, y-size, x+size, y+size), outline=(138, 43, 226))
                    else:
                        # Draw simple ellipse for other galaxy types
                        draw.ellipse((x-size, y-size, x+size, y+size), outline=(138, 43, 226))
                elif obj["type"] == "nebula":
                    color = (0, 191, 255)  # Blue for nebulae
                    # Draw a more irregular shape for nebulae
                    points = []
                    for i in range(8):
                        angle = i * (2 * math.pi / 8)
                        radius = size * (0.8 + 0.4 * random.random())
                        px = x + int(radius * math.cos(angle))
                        py = y + int(radius * math.sin(angle))
                        points.append((px, py))
                    draw.polygon(points, outline=(0, 191, 255))
                
                # Add label with name if available
                label = obj.get("name", obj["type"])
                confidence = f"{int(obj.get('confidence', 0.5) * 100)}%"
                
                # Draw the main label
                draw.text((x + size + 5, y - 15), label, fill=color, font=font)
                
                # Draw confidence
                draw.text((x + size + 5, y), confidence, fill=color, font=font)
                
                # Add scientific data if available
                y_offset = y + 15
                if "magnitude" in obj and obj["magnitude"] is not None:
                    draw.text((x + size + 5, y_offset), f"Mag: {obj['magnitude']:.2f}", fill=color, font=small_font)
                    y_offset += 12
                
                if "distance" in obj and obj["distance"] is not None:
                    distance_unit = obj.get("distance_unit", "")
                    draw.text((x + size + 5, y_offset), f"Dist: {obj['distance']} {distance_unit}", fill=color, font=small_font)
                    y_offset += 12
                
                if "redshift" in obj and obj["redshift"] is not None:
                    if obj["redshift"] > 0:
                        draw.text((x + size + 5, y_offset), f"Redshift: {obj['redshift']:.6f}", fill=color, font=small_font)
                    else:
                        draw.text((x + size + 5, y_offset), f"Blueshift: {abs(obj['redshift']):.6f}", fill=color, font=small_font)
            
            # Convert back to OpenCV format
            result_img = np.array(pil_img)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            
            return result_img
        except Exception as e:
            logger.error(f"Error in create_visualization: {e}")
            # Return original image if visualization fails
            return image
    
    def generate_summary(self, image_analysis, objects):
        """Generate a meaningful summary of the image analysis with scientific context."""
        try:
            # Count objects by type
            star_count = len([o for o in objects if o["type"] == "star"])
            galaxy_count = len([o for o in objects if o["type"] == "galaxy"])
            nebula_count = len([o for o in objects if o["type"] == "nebula"])
            
            # Count named objects
            named_stars = [o for o in objects if o["type"] == "star" and "name" in o and o["name"]]
            named_galaxies = [o for o in objects if o["type"] == "galaxy" and "name" in o and o["name"]]
            named_nebulae = [o for o in objects if o["type"] == "nebula" and "name" in o and o["name"]]
            
            # Generate summary text
            summary = {
                "image_type": image_analysis["image_type"],
                "total_objects": len(objects),
                "object_counts": {
                    "stars": star_count,
                    "galaxies": galaxy_count,
                    "nebulae": nebula_count
                },
                "named_objects": {
                    "stars": [o.get("name", "") for o in named_stars],
                    "galaxies": [o.get("name", "") for o in named_galaxies],
                    "nebulae": [o.get("name", "") for o in named_nebulae]
                },
                "brightness_level": "Low" if image_analysis["brightness"] < 80 else "Medium" if image_analysis["brightness"] < 150 else "High",
                "contrast_level": "Low" if image_analysis["contrast"] < 30 else "Medium" if image_analysis["contrast"] < 60 else "High",
                "complexity": image_analysis["complexity"],
                "dominant_color": image_analysis["dominant_color"]
            }
            
            # Add redshift/blueshift information if available
            if image_analysis["potential_redshift"]:
                summary["spectral_shift"] = "Potential redshift detected"
            elif image_analysis["potential_blueshift"]:
                summary["spectral_shift"] = "Potential blueshift detected"
            
            # Generate descriptive text based on the primary object type
            named_galaxy = next((o for o in objects if o["type"] == "galaxy" and "name" in o), None)
            
            if named_galaxy and "Andromeda" in named_galaxy["name"]:
                summary["description"] = f"This image shows the Andromeda Galaxy (M31), the nearest major galaxy to our Milky Way at a distance of {named_galaxy.get('distance', 2.537)} million light-years. It's a {named_galaxy.get('galaxy_type', 'spiral')} galaxy with an apparent magnitude of {named_galaxy.get('magnitude', 3.44):.2f}."
                
                if named_galaxy.get("redshift", 0) < 0:
                    summary["description"] += f" Andromeda has a blueshift of {abs(named_galaxy.get('redshift', 0.001001)):.6f}, indicating it's moving toward our galaxy at approximately 110 km/s."
                
                if star_count > 0:
                    summary["description"] += f" There are also approximately {star_count} stars visible in the foreground."
            
            elif named_galaxy:
                summary["description"] = f"This image shows the {named_galaxy['name']}, a {named_galaxy.get('galaxy_type', 'spiral')} galaxy located approximately {named_galaxy.get('distance', 'unknown')} {named_galaxy.get('distance_unit', 'light-years')} from Earth with an apparent magnitude of {named_galaxy.get('magnitude', 'unknown')}."
                
                if named_galaxy.get("redshift", 0) > 0:
                    summary["description"] += f" It has a redshift of {named_galaxy.get('redshift', 0):.6f}, indicating it's moving away from us."
                elif named_galaxy.get("redshift", 0) < 0:
                    summary["description"] += f" It has a blueshift of {abs(named_galaxy.get('redshift', 0)):.6f}, indicating it's moving toward us."
                
                if star_count > 0:
                    summary["description"] += f" There are also approximately {star_count} stars visible in the foreground."
            
            elif image_analysis["image_type"] == "galaxy":
                summary["description"] = f"This image appears to show an unidentified galaxy. The analysis detected {galaxy_count} galaxies, {star_count} stars, and {nebula_count} nebulae."
            
            elif galaxy_count > 0:
                summary["description"] = f"This image contains {galaxy_count} galaxies with {star_count} visible stars."
            
            elif star_count > 0 and nebula_count == 0:
                summary["description"] = f"This appears to be a star field containing approximately {star_count} visible stars."
            
            elif nebula_count > 0:
                summary["description"] = f"This image contains {nebula_count} nebulae and {star_count} stars."
            
            else:
                summary["description"] = f"This celestial image contains a mix of {star_count} stars, {galaxy_count} galaxies, and {nebula_count} nebulae."
            
            # Add information about notable objects
            if named_stars or named_galaxies or named_nebulae:
                summary["notable_objects"] = "Notable objects identified: "
                if named_galaxies:
                    summary["notable_objects"] += ", ".join([o.get("name", "") for o in named_galaxies[:3]]) + " (galaxies)"
                if named_stars:
                    if named_galaxies:
                        summary["notable_objects"] += "; "
                    summary["notable_objects"] += ", ".join([o.get("name", "") for o in named_stars[:3]]) + " (stars)"
                if named_nebulae:
                    if named_stars or named_galaxies:
                        summary["notable_objects"] += "; "
                    summary["notable_objects"] += ", ".join([o.get("name", "") for o in named_nebulae[:3]]) + " (nebulae)"
            
            return summary
        except Exception as e:
            logger.error(f"Error in generate_summary: {e}")
            # Return a basic summary if generation fails
            return {
                "image_type": image_analysis["image_type"],
                "total_objects": len(objects),
                "object_counts": {
                    "stars": len([o for o in objects if o["type"] == "star"]),
                    "galaxies": len([o for o in objects if o["type"] == "galaxy"]),
                    "nebulae": len([o for o in objects if o["type"] == "nebula"])
                },
                "brightness_level": "Medium",
                "contrast_level": "Medium",
                "complexity": image_analysis["complexity"],
                "description": "This image contains various celestial objects."
            }
    
    def process_image(self, image_path, output_image_path, output_json_path):
        """Process an astronomical image to detect celestial objects with scientific accuracy."""
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
            
            # Preprocess the image
            image_dict = self.preprocess_image(image)
            
            # Analyze the image content
            image_analysis = self.analyze_image_content(image_dict)
            
            # Detect stars
            stars = self.detect_stars(image_dict)
            
            # Detect galaxies
            galaxies = self.detect_galaxies(image_dict, image_analysis)
            
            # Detect nebulae
            nebulae = self.detect_nebulae(image_dict, image_analysis)
            
            # Combine all detected objects
            all_objects = stars + galaxies + nebulae
            
            # Generate summary
            summary = self.generate_summary(image_analysis, all_objects)
            
            # Create visualization
            visualization = self.create_visualization(image, all_objects)
            
            # Save the visualization
            cv2.imwrite(output_image_path, visualization)
            
            # Prepare the results
            results = {
                "objects": all_objects,
                "summary": summary,
                "image_analysis": image_analysis
            }
            
            # Save the results as JSON
            with open(output_json_path, 'w') as f:
                # Convert values to JSON-serializable types
                json_results = json.dumps(results, default=lambda o: 
                    float(o) if isinstance(o, (np.float32, np.float64)) else 
                    int(o) if isinstance(o, np.int64) else 
                    str(o) if isinstance(o, (np.bool_)) else o)
                f.write(json_results)
            
            return results
        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            raise

def main():
    """Main function to run the script from the command line."""
    if len(sys.argv) < 4:
        print("Usage: python simplified_detector.py <input_image_path> <output_image_path> <output_json_path>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    output_json_path = sys.argv[3]
    
    try:
        start_time = time.time()
        logger.info(f"Starting analysis of {input_image_path}")
        
        detector = CelestialObjectDetector()
        results = detector.process_image(input_image_path, output_image_path, output_json_path)
        
        end_time = time.time()
        logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
        
        summary = results["summary"]
        print(f"\nImage Analysis Complete:")
        print(f"Image Type: {summary['image_type']}")
        print(f"Total Objects Detected: {summary['total_objects']}")
        print(f"- Stars: {summary['object_counts']['stars']}")
        print(f"- Galaxies: {summary['object_counts']['galaxies']}")
        print(f"- Nebulae: {summary['object_counts']['nebulae']}")
        
        if summary['named_objects']['stars'] or summary['named_objects']['galaxies'] or summary['named_objects']['nebulae']:
            print("\nNotable Objects:")
            if summary['named_objects']['galaxies']:
                print(f"- Galaxies: {', '.join(summary['named_objects']['galaxies'])}")
            if summary['named_objects']['stars']:
                print(f"- Stars: {', '.join(summary['named_objects']['stars'])}")
            if summary['named_objects']['nebulae']:
                print(f"- Nebulae: {', '.join(summary['named_objects']['nebulae'])}")
        
        print(f"\nDescription: {summary['description']}")
        
        if 'spectral_shift' in summary:
            print(f"Spectral Analysis: {summary['spectral_shift']}")
        
        print(f"\nVisualization saved to: {output_image_path}")
        print(f"Results saved to: {output_json_path}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

