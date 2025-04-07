import os
import numpy as np
import pandas as pd
import requests
import cv2
import pickle
import logging
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from io import BytesIO
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AstronomicalTrainer:
    def __init__(self, data_dir="./training_data", model_dir="./models"):
        """Initialize the astronomical trainer"""
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        self.features = []
        self.labels = []
        self.object_names = []
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Known objects catalog for training
        self.known_objects = {
            "Andromeda Galaxy": {
                "type": "galaxy",
                "search_terms": ["M31", "NGC 224", "Andromeda Galaxy"],
                "visual_features": ["prominent bulge", "dusty spiral arms", "elliptical shape"]
            },
            "Whirlpool Galaxy": {
                "type": "galaxy",
                "search_terms": ["M51", "NGC 5194", "Whirlpool Galaxy"],
                "visual_features": ["clear spiral arms", "companion galaxy", "face-on orientation"]
            },
            "Triangulum Galaxy": {
                "type": "galaxy",
                "search_terms": ["M33", "NGC 598", "Triangulum Galaxy"],
                "visual_features": ["loose spiral structure", "bright star-forming regions", "diffuse appearance"]
            },
            "Sombrero Galaxy": {
                "type": "galaxy",
                "search_terms": ["M104", "NGC 4594", "Sombrero Galaxy"],
                "visual_features": ["prominent dust lane", "bright bulge", "edge-on orientation"]
            },
            "Milky Way": {
                "type": "galaxy",
                "search_terms": ["Milky Way", "Via Lactea"],
                "visual_features": ["dense star field", "dust lanes", "band across sky"]
            },
            "Pillars of Creation": {
                "type": "nebula",
                "search_terms": ["Pillars of Creation", "Eagle Nebula", "M16"],
                "visual_features": ["column structures", "reddish hue", "star formation region"]
            },
            "Crab Nebula": {
                "type": "nebula",
                "search_terms": ["Crab Nebula", "M1", "NGC 1952"],
                "visual_features": ["filamentary structure", "bluish glow", "supernova remnant"]
            },
            "Orion Nebula": {
                "type": "nebula",
                "search_terms": ["Orion Nebula", "M42", "NGC 1976"],
                "visual_features": ["bright center", "reddish glow", "dusty regions"]
            },
            "Pleiades": {
                "type": "star cluster",
                "search_terms": ["Pleiades", "Seven Sisters", "M45"],
                "visual_features": ["bright blue stars", "reflection nebulosity", "loose cluster"]
            }
        }
    
    def train_offline(self):
        """Train models offline without downloading new data"""
        logger.info("Training models with existing data...")
        
        # Check if we have any training data
        if not os.path.exists(self.data_dir):
            logger.warning("No training data directory found. Creating empty models.")
            self._create_empty_models()
            return True
            
        # Extract features from existing data
        self.extract_features()
        
        # Train models if we have features
        if len(self.features) > 0:
            success = self.train_model()
            if success:
                self.save_feature_extractor()
                logger.info("Models trained successfully with existing data")
                return True
        
        # If we don't have features or training failed, create empty models
        logger.warning("No features found or training failed. Creating empty models.")
        self._create_empty_models()
        return True
    
    def _create_empty_models(self):
        """Create empty models for fallback"""
        # Create a simple classifier
        self.classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        self.classifier.fit([[0, 0, 0, 0]], ["unknown"])
        
        # Create a simple object classifier
        self.object_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        self.object_classifier.fit([[0, 0, 0, 0]], ["unknown"])
        
        # Create a simple feature extractor
        def simple_extract_features(image):
            if image is None or image.size == 0:
                return None
                
            features = []
            
            # Convert to grayscale for some features
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Basic statistics
            for channel in cv2.split(image) if len(image.shape) == 3 else [gray]:
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.max(channel),
                    np.min(channel)
                ])
                
            return features
        
        # Save the models
        model_path = os.path.join(self.model_dir, "celestial_classifier.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
            
        object_model_path = os.path.join(self.model_dir, "object_classifier.pkl")
        with open(object_model_path, 'wb') as f:
            pickle.dump(self.object_classifier, f)
            
        extractor_path = os.path.join(self.model_dir, "feature_extractor.pkl")
        with open(extractor_path, 'wb') as f:
            pickle.dump(simple_extract_features, f)
            
        logger.info("Empty models created and saved")
    
    def download_sdss_images(self, limit=50):
        """Download images from SDSS using their API"""
        logger.info(f"Downloading {limit} images from SDSS...")
        
        # Create subdirectories for each object type
        for obj_name, obj_data in self.known_objects.items():
            obj_dir = os.path.join(self.data_dir, obj_data["type"], obj_name.replace(" ", "_"))
            os.makedirs(obj_dir, exist_ok=True)
            
            # Search for each known object
            for search_term in obj_data["search_terms"]:
                logger.info(f"Searching for {search_term}...")
                
                # Use SDSS SkyServer API to search for objects
                search_url = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
                
                # SQL query to find objects matching the search term
                # This is a simplified query - in a real implementation, you would use more specific criteria
                sql_query = f"""
                SELECT TOP {limit // len(obj_data['search_terms'])} 
                    p.objID, p.ra, p.dec, p.type, 
                    p.u, p.g, p.r, p.i, p.z,
                    p.petroRad_r
                FROM PhotoObj p
                JOIN SpecObj s ON p.objID = s.bestObjID
                WHERE s.class = 'GALAXY' 
                    AND (s.z BETWEEN 0.001 AND 0.1)
                    AND p.petroRad_r > 5
                ORDER BY p.petroRad_r DESC
                """
                
                # For nebulae and star clusters, use a different query
                if obj_data["type"] in ["nebula", "star cluster"]:
                    sql_query = f"""
                    SELECT TOP {limit // len(obj_data['search_terms'])} 
                        p.objID, p.ra, p.dec, p.type, 
                        p.u, p.g, p.r, p.i, p.z,
                        p.petroRad_r
                    FROM PhotoObj p
                    WHERE p.type = 3  -- Extended source
                        AND p.petroRad_r > 10
                    ORDER BY p.petroRad_r DESC
                    """
                
                # For the Milky Way, use a different approach - look for dense star fields
                if "Milky Way" in obj_name:
                    sql_query = f"""
                    SELECT TOP {limit // len(obj_data['search_terms'])} 
                        p.objID, p.ra, p.dec, 
                        COUNT(*) OVER (PARTITION BY CAST(p.ra/0.1 AS INT), CAST(p.dec/0.1 AS INT)) as star_density,
                        p.u, p.g, p.r, p.i, p.z
                    FROM PhotoObj p
                    WHERE p.type = 6  -- Star
                        AND p.ra BETWEEN 0 AND 360
                        AND p.dec BETWEEN -30 AND 30
                    ORDER BY star_density DESC
                    """
                
                try:
                    # Execute the query
                    params = {
                        "cmd": sql_query,
                        "format": "json"
                    }
                    
                    response = requests.get(search_url, params=params)
                    
                    if response.status_code == 200:
                        try:
                            results = response.json()
                            
                            # Process each result
                            if isinstance(results, list) and len(results) > 0:
                                for i, obj in enumerate(results):
                                    if i >= limit // len(obj_data['search_terms']):
                                        break
                                        
                                    # Extract RA and Dec
                                    if 'ra' in obj and 'dec' in obj:
                                        ra = obj['ra']
                                        dec = obj['dec']
                                        
                                        # Download image using SDSS cutout service
                                        self._download_sdss_cutout(ra, dec, obj_dir, f"{search_term}_{i}")
                                    
                            logger.info(f"Downloaded images for {search_term}")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON response for {search_term}")
                    else:
                        logger.error(f"Failed to query SDSS for {search_term}: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error downloading images for {search_term}: {e}")
        
        logger.info("Finished downloading SDSS images")
    
    def _download_sdss_cutout(self, ra, dec, save_dir, filename_prefix):
        """Download a cutout image from SDSS"""
        cutout_url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        
        params = {
            "ra": ra,
            "dec": dec,
            "scale": 0.4,
            "height": 512,
            "width": 512,
            "opt": "G"  # Grid overlay
        }
        
        try:
            response = requests.get(cutout_url, params=params)
            
            if response.status_code == 200:
                # Save the image
                img_path = os.path.join(save_dir, f"{filename_prefix}.jpg")
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded image to {img_path}")
                
                # Also save metadata
                metadata = {
                    "ra": ra,
                    "dec": dec,
                    "source": "SDSS DR16",
                    "download_date": datetime.now().isoformat()
                }
                
                metadata_path = os.path.join(save_dir, f"{filename_prefix}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"Failed to download cutout: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading cutout: {e}")
            return False
    
    def download_hubble_images(self):
        """Download images from Hubble Legacy Archive"""
        logger.info("Downloading images from Hubble Legacy Archive...")
        
        # For each known object, try to find Hubble images
        for obj_name, obj_data in self.known_objects.items():
            obj_dir = os.path.join(self.data_dir, obj_data["type"], obj_name.replace(" ", "_"))
            os.makedirs(obj_dir, exist_ok=True)
            
            # Use the first search term (usually the most common name)
            search_term = obj_data["search_terms"][0]
            
            # Hubble Legacy Archive API endpoint
            # Note: This is a simplified example. The actual HLA API is more complex.
            hla_url = "https://hla.stsci.edu/cgi-bin/hlaSIAP.cgi"
            
            params = {
                "POS": search_term,
                "SIZE": 0.2,
                "FORMAT": "image/jpeg"
            }
            
            try:
                response = requests.get(hla_url, params=params)
                
                if response.status_code == 200:
                    # Save the image
                    img_path = os.path.join(obj_dir, f"hubble_{search_term.replace(' ', '_')}.jpg")
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded Hubble image to {img_path}")
                else:
                    logger.warning(f"Failed to download Hubble image for {search_term}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error downloading Hubble image for {search_term}: {e}")
        
        logger.info("Finished downloading Hubble images")
    
    def download_additional_images(self):
        """Download additional images from other sources"""
        logger.info("Downloading additional images from other sources...")
        
        # NASA APOD API
        apod_url = "https://api.nasa.gov/planetary/apod"
        api_key = "DEMO_KEY"  # Replace with your NASA API key
        
        # For each known object, search NASA APOD
        for obj_name, obj_data in self.known_objects.items():
            obj_dir = os.path.join(self.data_dir, obj_data["type"], obj_name.replace(" ", "_"))
            os.makedirs(obj_dir, exist_ok=True)
            
            # Try each search term
            for search_term in obj_data["search_terms"]:
                params = {
                    "api_key": api_key,
                    "count": 5,
                    "thumbs": True
                }
                
                try:
                    response = requests.get(apod_url, params=params)
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Filter results that match our search term
                        matching_results = [
                            result for result in results 
                            if search_term.lower() in result.get('title', '').lower() or 
                               search_term.lower() in result.get('explanation', '').lower()
                        ]
                        
                        # Download matching images
                        for i, result in enumerate(matching_results):
                            if 'url' in result and (result['url'].endswith('.jpg') or result['url'].endswith('.png')):
                                img_url = result['url']
                                
                                # Download the image
                                img_response = requests.get(img_url)
                                
                                if img_response.status_code == 200:
                                    img_path = os.path.join(obj_dir, f"nasa_{search_term.replace(' ', '_')}_{i}.jpg")
                                    with open(img_path, 'wb') as f:
                                        f.write(img_response.content)
                                    logger.info(f"Downloaded NASA image to {img_path}")
                                    
                                    # Save metadata
                                    metadata = {
                                        "title": result.get('title'),
                                        "date": result.get('date'),
                                        "explanation": result.get('explanation'),
                                        "source": "NASA APOD",
                                        "download_date": datetime.now().isoformat()
                                    }
                                    
                                    metadata_path = os.path.join(obj_dir, f"nasa_{search_term.replace(' ', '_')}_{i}_metadata.json")
                                    with open(metadata_path, 'w') as f:
                                        json.dump(metadata, f, indent=2)
                    else:
                        logger.warning(f"Failed to query NASA APOD for {search_term}: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error downloading NASA images for {search_term}: {e}")
        
        logger.info("Finished downloading additional images")
    
    def extract_features(self):
        """Extract features from downloaded images"""
        logger.info("Extracting features from images...")
        
        self.features = []
        self.labels = []
        self.object_names = []
        
        # Process each object type directory
        for obj_type in os.listdir(self.data_dir):
            type_dir = os.path.join(self.data_dir, obj_type)
            
            if not os.path.isdir(type_dir):
                continue
                
            # Process each object directory
            for obj_name in os.listdir(type_dir):
                obj_dir = os.path.join(type_dir, obj_name)
                
                if not os.path.isdir(obj_dir):
                    continue
                    
                # Process each image in the object directory
                for filename in os.listdir(obj_dir):
                    if not filename.endswith(('.jpg', '.png', '.fits')):
                        continue
                        
                    filepath = os.path.join(obj_dir, filename)
                    
                    try:
                        # Load and process image
                        if filename.endswith('.fits'):
                            # Process FITS file
                            with fits.open(filepath) as hdul:
                                image_data = hdul[0].data
                                # Convert to 8-bit if necessary
                                if image_data.dtype != np.uint8:
                                    image_data = self._normalize_fits(image_data)
                        else:
                            # Process regular image
                            image_data = cv2.imread(filepath)
                            if image_data is None:
                                logger.warning(f"Failed to load image: {filepath}")
                                continue
                                
                            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                        
                        # Extract features
                        image_features = self._extract_image_features(image_data)
                        
                        if image_features is not None:
                            # Add to dataset
                            self.features.append(image_features)
                            self.labels.append(obj_type)
                            self.object_names.append(obj_name.replace("_", " "))
                            
                            logger.info(f"Extracted features from {filepath}")
                        else:
                            logger.warning(f"Failed to extract features from {filepath}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.object_names = np.array(self.object_names)
        
        logger.info(f"Extracted features from {len(self.features)} images")
        
        # Save the extracted features
        features_path = os.path.join(self.model_dir, "extracted_features.npz")
        np.savez(
            features_path, 
            features=self.features, 
            labels=self.labels, 
            object_names=self.object_names
        )
        logger.info(f"Saved extracted features to {features_path}")
    
    def _normalize_fits(self, data):
        """Normalize FITS data to 8-bit range"""
        data = data.astype(float)
        data -= np.min(data)
        if np.max(data) > 0:
            data /= np.max(data)
        data *= 255
        return data.astype(np.uint8)
    
    def _extract_image_features(self, image):
        """Extract features from an image"""
        if image is None or image.size == 0:
            return None
            
        features = []
        
        # Convert to grayscale for some features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 1. Basic statistics
        for channel in cv2.split(image) if len(image.shape) == 3 else [gray]:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.max(channel),
                np.min(channel)
            ])
            
        # 2. Histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / np.sum(hist)  # Normalize
        features.extend(hist)
        
        # 3. Edge features
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]))
        
        # 4. Blob features
        params = cv2.SimpleBlobDetector_Params()
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        features.append(len(keypoints))
        
        # 5. Color distribution
        if len(image.shape) == 3:
            r, g, b = cv2.split(image)
            r_mean, g_mean, b_mean = np.mean(r)/255.0, np.mean(g)/255.0, np.mean(b)/255.0
            total = r_mean + g_mean + b_mean
            if total > 0:
                r_ratio, g_ratio, b_ratio = r_mean/total, g_mean/total, b_mean/total
            else:
                r_ratio, g_ratio, b_ratio = 0.33, 0.33, 0.33
            
            features.extend([r_ratio, g_ratio, b_ratio])
        
        # 6. Star detection (for astronomical images)
        try:
            mean, median, std = sigma_clipped_stats(gray, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
            sources = daofind(gray - median)
            
            if sources is not None:
                features.append(len(sources))
            else:
                features.append(0)
        except:
            features.append(0)
        
        return features
    
    def train_model(self):
        """Train the classifier on extracted features"""
        if len(self.features) == 0:
            logger.error("No features extracted. Run extract_features() first.")
            return False
            
        logger.info("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            self.features, self.labels, self.object_names, test_size=0.3, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(self.model_dir, "celestial_classifier.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
            
        logger.info(f"Model trained and saved as {model_path}")
        
        # Train a second classifier for specific object names
        object_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        object_classifier.fit(X_train, names_train)
        
        # Evaluate object classifier
        names_pred = object_classifier.predict(X_test)
        accuracy = np.mean(names_pred == names_test)
        logger.info(f"Object name classification accuracy: {accuracy:.2f}")
        
        # Save object classifier
        object_model_path = os.path.join(self.model_dir, "object_classifier.pkl")
        with open(object_model_path, 'wb') as f:
            pickle.dump(object_classifier, f)
            
        logger.info(f"Object classifier saved as {object_model_path}")
        
        return True
    
    def save_feature_extractor(self):
        """Save the feature extraction function"""
        extractor_path = os.path.join(self.model_dir, "feature_extractor.pkl")
        with open(extractor_path, 'wb') as f:
            pickle.dump(self._extract_image_features, f)
            
        logger.info(f"Feature extractor saved as {extractor_path}")
        
    def load_or_train(self):
        """Load existing models or train new ones if they don't exist"""
        model_path = os.path.join(self.model_dir, "celestial_classifier.pkl")
        object_model_path = os.path.join(self.model_dir, "object_classifier.pkl")
        features_path = os.path.join(self.model_dir, "extracted_features.npz")
        
        if os.path.exists(model_path) and os.path.exists(object_model_path) and os.path.exists(features_path):
            # Load existing models and features
            logger.info("Loading existing models and features...")
            
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
                
            with open(object_model_path, 'rb') as f:
                self.object_classifier = pickle.load(f)
                
            data = np.load(features_path)
            self.features = data['features']
            self.labels = data['labels']
            self.object_names = data['object_names']
            
            logger.info("Models and features loaded successfully")
            return True
        else:
            # Download data and train new models
            logger.info("No existing models found. Downloading data and training new models...")
            
            # Download training data
            self.download_sdss_images()
            self.download_hubble_images()
            self.download_additional_images()
            
            # Extract features and train models
            self.extract_features()
            success = self.train_model()
            
            if success:
                self.save_feature_extractor()
                logger.info("New models trained successfully")
                return True
            else:
                logger.error("Failed to train new models")
                return False

# Main function to run the trainer
def main():
    trainer = AstronomicalTrainer()
    trainer.train_offline()

if __name__ == "__main__":
    main()

