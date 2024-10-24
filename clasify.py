import cv2
import os

# Paths
pos_samples_path = "path/to/positive_samples/"  # Gambar positif
neg_samples_path = "path/to/negative_samples/"  # Gambar negatif
cascade_output_path = "path/to/output/cascade.xml"  # Output XML

# Create a list of positive samples
pos_samples = [os.path.join(pos_samples_path, f) for f in os.listdir(pos_samples_path) if f.endswith('.jpg')]
pos_samples_count = len(pos_samples)

# Create a list of negative samples
neg_samples = [os.path.join(neg_samples_path, f) for f in os.listdir(neg_samples_path) if f.endswith('.jpg')]
neg_samples_count = len(neg_samples)

# Generate .vec file from positive samples
os.system(f"opencv_createsamples -info positives.info -num {pos_samples_count} -w 24 -h 24 -vec positives.vec")

# Train the cascade classifier
os.system(f"opencv_traincascade -data data -vec positives.vec -bg negatives.txt -numPos {int(pos_samples_count * 0.9)} -numNeg {neg_samples_count} -numStages 10 -featureType HAAR -w 24 -h 24")
