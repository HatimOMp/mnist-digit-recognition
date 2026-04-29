import cv2
import numpy as np
import tensorflow as tf


class DigitExtractionPipeline:
    """
    End-to-end pipeline for extracting and recognizing digits
    from document images (receipts, invoices, forms).
    """

    def __init__(self, model_path="mnist_model.keras"):
        self.model = tf.keras.models.load_model(model_path)
        self.confidence_threshold = 0.7

    # ── Step 1: Preprocessing ────────────────────────────────────────
    def preprocess(self, image):
        """Convert to grayscale, denoise, and threshold."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Denoise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding — handles uneven lighting
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Morphological operations to clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return gray, thresh, cleaned

    # ── Step 2: Digit Detection ───────────────────────────────────────
    def detect_digits(self, cleaned, original):
        """Find and extract individual digit regions."""
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        digit_regions = []
        image_h, image_w = original.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by minimum size
            if w < 8 or h < 15:
                continue

            # Filter by maximum size
            if w > image_w * 0.15 or h > image_h * 0.08:
                continue

            # Filter by aspect ratio — digits are taller than wide
            aspect_ratio = w / h
            if aspect_ratio > 0.9:
                continue

            # Filter by solidity — ratio of contour area to bounding box
            area = cv2.contourArea(contour)
            bbox_area = w * h
            solidity = area / bbox_area if bbox_area > 0 else 0
            if solidity < 0.2 or solidity > 0.95:
                continue

            # Filter by minimum area
            if area < 80:
                continue

            digit_regions.append((x, y, w, h))

        # Sort top-to-bottom, left-to-right
        digit_regions = sorted(
            digit_regions, key=lambda r: (r[1] // 20, r[0])
        )

        return digit_regions

    # ── Step 3: Classification ────────────────────────────────────────
    def classify_digit(self, image, region):
        """Classify a single digit region."""
        x, y, w, h = region

        # Add padding around digit
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        digit_img = image[y1:y2, x1:x2]

        # Convert to grayscale if needed
        if len(digit_img.shape) == 3:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

        # Invert if background is dark
        if np.mean(digit_img) > 127:
            digit_img = cv2.bitwise_not(digit_img)

        # Resize to 28x28
        digit_img = cv2.resize(digit_img, (28, 28))

        # Normalize
        digit_img = digit_img.astype("float32") / 255.0
        digit_img = digit_img[np.newaxis, ..., np.newaxis]

        # Predict
        predictions = self.model.predict(digit_img, verbose=0)[0]
        predicted = np.argmax(predictions)
        confidence = float(predictions[predicted])

        return predicted, confidence, predictions

    # ── Step 4: Annotate image ────────────────────────────────────────
    def annotate_image(self, image, regions, predictions):
        """Draw bounding boxes and predictions on image."""
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h), (digit, confidence, _) in zip(regions, predictions):
            color = (0, 200, 0) if confidence >= self.confidence_threshold else (0, 100, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{digit} ({confidence:.0%})"
            cv2.putText(
                annotated, label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1
            )

        return annotated

    # ── Step 5: Structure output ──────────────────────────────────────
    def structure_output(self, regions, predictions, image_shape):
        """Group digits by line and reconstruct numbers."""
        if not regions:
            return []

        image_h = image_shape[0]
        line_height = image_h * 0.04

        lines = []
        current_line = [(regions[0], predictions[0])]

        for region, pred in zip(regions[1:], predictions[1:]):
            x, y, w, h = region
            prev_y = current_line[-1][0][1]

            if abs(y - prev_y) <= line_height:
                current_line.append((region, pred))
            else:
                lines.append(current_line)
                current_line = [(region, pred)]
        lines.append(current_line)

        structured = []
        for line in lines:
            digits = [str(pred[0]) for _, pred in line
                      if pred[1] >= self.confidence_threshold]
            confidences = [pred[1] for _, pred in line
                           if pred[1] >= self.confidence_threshold]
            if digits:
                number = "".join(digits)
                avg_confidence = np.mean(confidences)
                structured.append({
                    "number": number,
                    "digits": digits,
                    "avg_confidence": avg_confidence
                })

        return structured

    # ── Full pipeline ─────────────────────────────────────────────────
    def run(self, image):
        """Run the full extraction pipeline on an image."""
        gray, thresh, cleaned = self.preprocess(image)
        regions = self.detect_digits(cleaned, image)

        if not regions:
            return {
                "annotated": image,
                "gray": gray,
                "thresh": thresh,
                "regions": [],
                "predictions": [],
                "structured": [],
                "digit_count": 0
            }

        predictions = [self.classify_digit(image, r) for r in regions]
        annotated = self.annotate_image(image, regions, predictions)
        structured = self.structure_output(regions, predictions, image.shape)

        return {
            "annotated": annotated,
            "gray": gray,
            "thresh": thresh,
            "regions": regions,
            "predictions": predictions,
            "structured": structured,
            "digit_count": len(regions)
        }