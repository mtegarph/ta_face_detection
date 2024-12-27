import numpy as np
import xml.etree.ElementTree as ET

class HaarFeature:
    def __init__(self):
        self.feature_type = None
        self.position = None
        self.width = None
        self.height = None
        self.threshold = None
        self.left_val = None
        self.right_val = None
        self.weight = None

class PureHaarCascadeClassifier:
    def __init__(self, cascade_path):
        self.features = []
        self.stages = []
        self.load_cascade(cascade_path)
    
    def compute_integral_image(self, image):
        height, width = image.shape
        integral = np.zeros((height + 1, width + 1), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                integral[y + 1, x + 1] = (image[y, x] + 
                                        integral[y, x + 1] + 
                                        integral[y + 1, x] - 
                                        integral[y, x])
        return integral
    
    def compute_variance_integral(self, image):
        height, width = image.shape
        squared = np.square(image.astype(np.float32))
        variance_integral = np.zeros((height + 1, width + 1), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                variance_integral[y + 1, x + 1] = (squared[y, x] + 
                                                 variance_integral[y, x + 1] + 
                                                 variance_integral[y + 1, x] - 
                                                 variance_integral[y, x])
        return variance_integral
    
    def evaluate_feature(self, integral_image, variance_integral, x, y, scale, feature):
        # Calculate the feature value using integral image
        rect_sum = 0
        window_size = int(24 * scale)  # Base window size is 24x24
        
        # Normalize the feature response using local variance
        window_area = window_size * window_size
        window_sum = (integral_image[y + window_size, x + window_size] -
                     integral_image[y + window_size, x] -
                     integral_image[y, x + window_size] +
                     integral_image[y, x])
        window_sqsum = (variance_integral[y + window_size, x + window_size] -
                       variance_integral[y + window_size, x] -
                       variance_integral[y, x + window_size] +
                       variance_integral[y, x])
        
        variance = window_sqsum * window_area - window_sum * window_sum
        if variance > 0:
            std_dev = np.sqrt(variance)
        else:
            return 0
        
        # Evaluate rectangles
        for rect in feature.rects:
            rx = int(x + rect.x * scale)
            ry = int(y + rect.y * scale)
            rw = int(rect.width * scale)
            rh = int(rect.height * scale)
            
            rect_sum += (integral_image[ry + rh, rx + rw] -
                        integral_image[ry + rh, rx] -
                        integral_image[ry, rx + rw] +
                        integral_image[ry, rx]) * rect.weight
        
        return rect_sum / std_dev
    
    def detectMultiScale(self, image, scale_factor=1.1, min_neighbors=3):
        height, width = image.shape
        integral = self.compute_integral_image(image)
        variance_integral = self.compute_variance_integral(image)
        
        min_size = 24  # Minimum window size
        detections = []
        scale = 1.0
        
        while min_size * scale < min(height, width):
            window_size = int(min_size * scale)
            step = max(2, int(window_size * 0.1))  # Sliding step
            
            for y in range(0, height - window_size, step):
                for x in range(0, width - window_size, step):
                    passes_stages = True
                    
                    # Check each stage
                    for stage in self.stages:
                        stage_sum = 0
                        
                        for feature in stage.features:
                            feature_val = self.evaluate_feature(
                                integral, variance_integral, x, y, scale, feature)
                            
                            if feature_val < feature.threshold:
                                stage_sum += feature.left_val
                            else:
                                stage_sum += feature.right_val
                        
                        if stage_sum < stage.threshold:
                            passes_stages = False
                            break
                    
                    if passes_stages:
                        detections.append((x, y, window_size, window_size))
            
            scale *= scale_factor
        
        # Group overlapping detections
        if len(detections) > 0:
            detections = self.group_rectangles(detections, min_neighbors)
        
        return detections
    
    def group_rectangles(self, rectangles, min_neighbors):
        if len(rectangles) == 0:
            return []
        
        # Convert rectangles to numpy array
        rectangles = np.array(rectangles)
        
        # Group overlapping rectangles
        groups = []
        for rect in rectangles:
            matched = False
            for group in groups:
                if self.is_overlap(rect, group[0]):
                    group.append(rect)
                    matched = True
                    break
            if not matched:
                groups.append([rect])
        
        # Filter groups and compute average rectangles
        result = []
        for group in groups:
            if len(group) >= min_neighbors:
                mean_rect = np.mean(group, axis=0)
                result.append(tuple(map(int, mean_rect)))
        
        return result
    
    def is_overlap(self, rect1, rect2):
        # Calculate overlap ratio between two rectangles
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate intersection
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        
        if w * h == 0:
            return False
        
        # Calculate overlap ratio
        intersection = w * h
        union = w1 * h1 + w2 * h2 - intersection
        overlap = intersection / union
        
        return overlap > 0.3
    def __init__(self, cascade_path):
        """
        Initialize Haar Cascade Classifier with XML cascade file
        
        Args:
            cascade_path (str): Path to XML cascade file
        """
        self.stages = []
        self.load_cascade(cascade_path)
    
    def load_cascade(self, cascade_path):
        """
        Load cascade from XML file
        
        Args:
            cascade_path (str): Path to XML cascade file
        """
        try:
            tree = ET.parse(cascade_path)
            root = tree.getroot()
            
            # Parse stages
            for stage_elem in root.findall('.//stage'):
                features = []
                stage_threshold = float(stage_elem.find('stage_threshold').text)
                
                # Parse features in stage
                for feature_elem in stage_elem.findall('.//feature'):
                    rects = feature_elem.findall('rects/rect')
                    
                    # Parse feature type and parameters
                    feature_type = len(rects)  # Number of rectangles determines feature type
                    position = (int(rects[0].text.split()[0]), int(rects[0].text.split()[1]))
                    width = int(rects[0].text.split()[2])
                    height = int(rects[0].text.split()[3])
                    threshold = float(feature_elem.find('threshold').text)
                    left_val = float(feature_elem.find('left_val').text)
                    right_val = float(feature_elem.find('right_val').text)
                    
                    feature = HaarFeature(
                        feature_type, position, width, height,
                        threshold, left_val, right_val
                    )
                    features.append(feature)
                
                stage = HaarStage(features, stage_threshold)
                self.stages.append(stage)
        
        except Exception as e:
            print(f"Error loading cascade file: {e}")
            self.stages = []
    
    def compute_integral_image(self, image):
        """
        Compute integral image for efficient feature computation
        
        Args:
            image (numpy.ndarray): Input grayscale image
        
        Returns:
            numpy.ndarray: Integral image
        """
        height, width = image.shape
        integral = np.zeros((height + 1, width + 1), dtype=np.float32)
        
        # Compute integral image
        for y in range(height):
            for x in range(width):
                integral[y + 1, x + 1] = (image[y, x] +
                                        integral[y, x + 1] +
                                        integral[y + 1, x] -
                                        integral[y, x])
        
        return integral
    
    def compute_feature(self, integral_image, feature, scale, x, y):
        """
        Compute value of Haar-like feature
        
        Args:
            integral_image (numpy.ndarray): Integral image
            feature (HaarFeature): Haar feature to compute
            scale (float): Scale factor
            x (int): X coordinate
            y (int): Y coordinate
        
        Returns:
            float: Feature value
        """
        # Scale feature dimensions
        width = int(feature.width * scale)
        height = int(feature.height * scale)
        pos_x = int(x + feature.position[0] * scale)
        pos_y = int(y + feature.position[1] * scale)
        
        # Compute rectangle sum using integral image
        def rect_sum(rx, ry, rw, rh):
            return (integral_image[ry + rh, rx + rw] -
                    integral_image[ry + rh, rx] -
                    integral_image[ry, rx + rw] +
                    integral_image[ry, rx])
        
        # Compute feature value based on type
        if feature.feature_type == 2:  # Two-rectangle feature
            white = rect_sum(pos_x, pos_y, width//2, height)
            black = rect_sum(pos_x + width//2, pos_y, width//2, height)
            return white - black
        elif feature.feature_type == 3:  # Three-rectangle feature
            white = rect_sum(pos_x + width//3, pos_y, width//3, height)
            black = rect_sum(pos_x, pos_y, width//3, height) + rect_sum(pos_x + 2*width//3, pos_y, width//3, height)
            return white - black
        else:  # Four-rectangle feature
            white = rect_sum(pos_x, pos_y, width//2, height//2) + rect_sum(pos_x + width//2, pos_y + height//2, width//2, height//2)
            black = rect_sum(pos_x + width//2, pos_y, width//2, height//2) + rect_sum(pos_x, pos_y + height//2, width//2, height//2)
            return white - black
    
    def detectMultiScale(self, image, scale_factor=1.1, min_neighbors=3):
        """
        Detect objects using the Haar Cascade Classifier
        
        Args:
            image (numpy.ndarray): Input grayscale image
            scale_factor (float): Scale factor between subsequent scans
            min_neighbors (int): Minimum neighbors threshold
        
        Returns:
            list: List of detected objects as (x, y, w, h) rectangles
        """
        height, width = image.shape
        integral = self.compute_integral_image(image)
        
        # Store detected faces
        detections = []
        
        # Scan image at multiple scales
        scale = 1
        minSsize = 24  # Minimum detection window size
        
        while minSsize * scale < min(height, width):
            step = int(2 * scale)  # Step size
            window_w = int(minSsize * scale)
            window_h = int(minSsize * scale)
            
            for y in range(0, height - window_h, step):
                for x in range(0, width - window_w, step):
                    # Check if window passes all stages
                    passed = True
                    
                    for stage in self.stages:
                        stage_sum = 0
                        
                        for feature in stage.features:
                            feature_val = self.compute_feature(integral, feature, scale, x, y)
                            if feature_val < feature.threshold:
                                stage_sum += feature.left_val
                            else:
                                stage_sum += feature.right_val
                        
                        if stage_sum < stage.threshold:
                            passed = False
                            break
                    
                    if passed:
                        detections.append((x, y, window_w, window_h))
            
            scale *= scale_factor
        
        # Group overlapping detections
        if len(detections) > 0:
            detections = self.group_rectangles(detections, min_neighbors)
        
        return detections
    
    def group_rectangles(self, rectangles, min_neighbors):
        """
        Group overlapping rectangles
        
        Args:
            rectangles (list): List of rectangles as (x, y, w, h)
            min_neighbors (int): Minimum neighbors threshold
        
        Returns:
            list: Filtered list of rectangles
        """
        if len(rectangles) == 0:
            return []
        
        # Group overlapping rectangles
        groups = []
        for rect in rectangles:
            matched = False
            for group in groups:
                if self.is_overlap(rect, group[0]):
                    group.append(rect)
                    matched = True
                    break
            if not matched:
                groups.append([rect])
        
        # Filter groups and compute average rectangles
        result = []
        for group in groups:
            if len(group) >= min_neighbors:
                mean_rect = np.mean(group, axis=0)
                result.append(tuple(int(v) for v in mean_rect))
        
        return result
    
    def is_overlap(self, rect1, rect2):
        """
        Check if two rectangles overlap
        
        Args:
            rect1 (tuple): First rectangle (x, y, w, h)
            rect2 (tuple): Second rectangle (x, y, w, h)
        
        Returns:
            bool: True if rectangles overlap
        """
        overlap_threshold = 0.3
        
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate overlap area
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = x_overlap * y_overlap
        
        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - overlap_area
        
        return overlap_area / union_area > overlap_threshold
    
#     import numpy as np
# class HaarFeature:
#     def __init__(self, feature_type, position, width, height, threshold, left_val, right_val):
#         self.feature_type = feature_type
#         self.position = position
#         self.width = width
#         self.height = height
#         self.threshold = threshold
#         self.left_val = left_val
#         self.right_val = right_val
#         self.rects = []  # Initialize to store rectangle definitions

# class HaarStage:
#     def __init__(self, features, stage_threshold):
#         self.features = features
#         self.threshold = stage_threshold

# class PureHaarCascadeClassifier:
#     def __init__(self, cascade_path):
#         self.stages = []
#         self.load_cascade(cascade_path)

#     def load_cascade(self, cascade_path):
#         try:
#             tree = ET.parse(cascade_path)
#             root = tree.getroot()

#             for stage_elem in root.findall('.//stage'):
#                 features = []
#                 stage_threshold = float(stage_elem.find('stage_threshold').text)

#                 for feature_elem in stage_elem.findall('.//feature'):
#                     rects = feature_elem.findall('rects/rect')
#                     feature_type = len(rects)
#                     position = (int(rects[0](citation_0).text.split()[0](citation_0)), int(rects[0](citation_0).text.split()[1](citation_1)))
#                     width = int(rects[0](citation_0).text.split()[2](citation_2))
#                     height = int(rects[0](citation_0).text.split()[3](citation_3))
#                     threshold = float(feature_elem.find('threshold').text)
#                     left_val = float(feature_elem.find('left_val').text)
#                     right_val = float(feature_elem.find('right_val').text)

#                     feature = HaarFeature(feature_type, position, width, height, threshold, left_val, right_val)
#                     feature.rects = [self.parse_rect(rect) for rect in rects]  # Add rectangles
#                     features.append(feature)

#                 stage = HaarStage(features, stage_threshold)
#                 self.stages.append(stage)

#         except Exception as e:
#             print(f"Error loading cascade file: {e}")
#             self.stages = []

#     def parse_rect(self, rect_elem):
#         # Parse individual rectangle elements
#         x, y, width, height = map(int, rect_elem.text.split())
#         return {'x': x, 'y': y, 'width': width, 'height': height}

#     def compute_integral_image(self, image):
#         height, width = image.shape
#         integral = np.zeros((height + 1, width + 1), dtype=np.float32)

#         for y in range(height):
#             for x in range(width):
#                 integral[y + 1, x + 1] = (image[y, x] +
#                                            integral[y, x + 1] +
#                                            integral[y + 1, x] -
#                                            integral[y, x])
#         return integral

#     def evaluate_feature(self, integral_image, x, y, scale, feature):
#         rect_sum = 0
#         for rect in feature.rects:
#             rx = int(x + rect['x'] * scale)
#             ry = int(y + rect['y'] * scale)
#             rw = int(rect['width'] * scale)
#             rh = int(rect['height'] * scale)
#             rect_sum += (integral_image[ry + rh, rx + rw] -
#                           integral_image[ry + rh, rx] -
#                           integral_image[ry, rx + rw] +
#                           integral_image[ry, rx]) * rect['weight']

#         return rect_sum

#     def detectMultiScale(self, image, scale_factor=1.1, min_neighbors=3):
#         height, width = image.shape
#         integral = self.compute_integral_image(image)

#         detections = []
#         scale = 1.0
#         min_size = 24

#         while min_size * scale < min(height, width):
#             window_size = int(min_size * scale)
#             step = max(2, int(window_size * 0.1))

#             for y in range(0, height - window_size, step):
#                 for x in range(0, width - window_size, step):
#                     passes_stages = True

#                     for stage in self.stages:
#                         stage_sum = 0

#                         for feature in stage.features:
#                             feature_val = self.evaluate_feature(integral, x, y, scale, feature)

#                             if feature_val < feature.threshold:
#                                 stage_sum += feature.left_val
#                             else:
#                                 stage_sum += feature.right_val

#                         if stage_sum < stage.threshold:
#                             passes_stages = False
#                             break

#                     if passes_stages:
#                         detections.append((x, y, window_size, window_size))

#             scale *= scale_factor

#         if len(detections) > 0:
#             detections = self.group_rectangles(detections, min_neighbors)

#         return detections

#     def group_rectangles(self, rectangles, min_neighbors):
#         if len(rectangles) == 0:
#             return []

#         groups = []
#         for rect in rectangles:
#             matched = False
#             for group in groups:
#                 if self.is_overlap(rect, group[0](citation_0)):
#                     group.append(rect)
#                     matched = True
#                     break
#             if not matched:
#                 groups.append([rect])

#         result = []
#         for group in groups:
#             if len(group) >= min_neighbors:
#                 mean_rect = np.mean(group, axis=0)
#                 result.append(tuple(map(int, mean_rect)))

#         return result

#     def is_overlap(self, rect1, rect2):
#         x1, y1, w1, h1 = rect1
#         x2, y2, w2, h2 = rect2

#         x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
#         y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
#         overlap_area = x_overlap * y_overlap

#         area1 = w1 * h1
#         area2 = w2 * h2
#         union_area = area1 + area2 - overlap_area

#         return overlap_area / union_area > 0.3