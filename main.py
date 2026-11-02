from datetime import datetime
from typing import List, Dict, Any

# =======================================
#   1. DATA INFRASTRUCTURE & UTILITIES
# =======================================

class Bounding_Box:
    """This is the box that trucks will be framed in"""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    # checking to see if multiple bounding boxes are overlapped
    def intersection(self, other: 'Bounding_Box') -> bool:
        # Placeholder for overlapping geometric calculation
        return True

    # should help us distinguish boxes if they are overlapped
    def center(self) -> tuple:
        return (self.x + self.width / 2, self.y + self.height / 2)


class Image_Payload:
    """Packages a raw image with information we need for processing"""
    def __init__(self, imageID: str, rawBytes: bytes, timestamp: datetime, metadata: Dict[str, Any]):
        self.imageID = imageID
        self.rawBytes = rawBytes
        self.timestamp = timestamp
        self.metadata = metadata

    def getMetaData(self, key: str) -> Any:
        return self.metadata.get(key)

    def updateMetadata(self, key: str, value: Any) -> bool:
        self.metadata[key] = value
        return True

    def getSize(self) -> int:
        return len(self.rawBytes)

# =======================================
#   2. INFERENCE & PROCESSING PIPELINE
# =======================================

class Inference_Result:
    """Captures the outputs from the model"""
    def __init__(self, detectedObjects: List[str], confidenceScores: List[float], boundingBoxes: List[Bounding_Box], timestamp: datetime):
        self.detectedObjects = detectedObjects
        self.confidenceScores = confidenceScores
        self.boundingBoxes = boundingBoxes  # Composition: Has many Bounding Boxes
        self.timestamp = timestamp

    # ================================
    #   detectedObjects getter/setter
    # ================================
    @property
    def detectedObjects(self) -> List[str]:
        return self._detectedObjects

    @detectedObjects.setter
    def detectedObjects(self, detObjList):
        """Setter checks if detectedObjects is in list form"""
        if not isinstance(detObjList, list):
            raise TypeError("Detected Objects is not in List form.")
        self._detectedObjects = detObjList   

    # =================================
    #   confidenceScores getter/setter
    # =================================
    @property
    def confidenceScores(self) -> List[float]:
        return self._confidenceScores
    
    @confidenceScores.setter
    def confidenceScores(self, confScoreList):
        """Setter checks if confidenceScores is in list form"""
        if not isinstance(confScoreList, list):
            raise TypeError("Confidence Scores is not in List form.")
        self._confidenceScores = confScoreList

    # ==============================
    #   boundingBoxes getter/setter
    # ==============================
    @property
    def boundingBoxes(self) -> List[Bounding_Box]:
        return self._boundingBoxes
    
    @boundingBoxes.setter
    def boundingBoxes(self, boundBoxList):
        """Setter checks if boundingBoxes is in list form"""
        if not isinstance(boundBoxList, list):
            raise TypeError("Bounding Boxes are not in list form.")
        self._boundingBoxes = boundBoxList

    # ==========================
    #   timestamp getter/setter
    # ==========================
    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, dtStamp):
        """Setter checks if timestamp is in datetime form"""
        if not isinstance(dtStamp, datetime):
            raise TypeError("Timestamp is not in datetime form.")
        self._timestamp = dtStamp

    # ==================
    #   Utility Getters
    # ==================
    @property
    def getTopConfidence(self) -> float:
        """Returns top confidence score"""
        return max(self._confidenceScores) if self._confidenceScores else 0.0
    @property
    def hasDetections(self) -> bool:
        """Returns boolean representation of whether a detectedObjects list exist"""
        return len(self._detectedObjects) > 0

    @property
    def getDetectionCount(self) -> int:
        """Returns how many detections are in the detectedObjects list"""
        return len(self._detectedObjects)

class Image_Process_Engine:
    """Preprocesses image and passes to the Truck Counting Engine"""
    def __init__(self, processingMode: str, lastProcessedAt: datetime):
        self.processingMode = processingMode
        self.lastProcessedAt = lastProcessedAt

    # ================================
    #   Processing Mode getter/setter
    # ================================
    @property
    def processingMode(self) -> str:
        return self._processingMode

    @processingMode.setter
    def processingMode(self, pMode):
        if not isinstance(pMode, str):
            raise TypeError("Processing Mode must be of type string.")
        self._processingMode = pMode

    # ================================
    #   lastProcessedAt getter/setter
    # ================================
    @property
    def lastProcessedAt(self) -> datetime:
        return self._lastProcessedAt

    @lastProcessedAt.setter
    def lastProcessedAt(self, lastPTime):
        if not isinstance(lastPTime, datetime):
            raise TypeError("lastProcessedAt must be of type datetime.")
        self._lastProcessedAt = lastPTime

    # ==================
    #   Utility Getters
    # ==================
    def preprocess(self, imagePayload: Image_Payload) -> Image_Payload:
        return imagePayload

    def postprocess(self, report: 'Facility_Metrics_Report') -> 'Facility_Metrics_Report':
        return report

class Truck_Detection_Model(Image_Process_Engine):
    """This is where our model lives"""
    def __init__(self, modelVersion: str, confidenceThreshold: float, processingMode: str, lastProcessedAt: datetime):
        super().__init__(processingMode, lastProcessedAt)
        self.modelVersion = modelVersion
        self.confidenceThreshold = confidenceThreshold

    # =============================
    #   modelVersion setter/getter
    # =============================
    @property
    def modelVersion(self) -> str:
        return self._modelVersion

    @modelVersion.setter
    def modelVersion(self, mVersion):
        if not isinstance(mVersion, str):
            raise TypeError("Model Version must be of type string.")
        self._modelVersion = mVersion
    
    # ====================================
    #   confidenceThreshold setter/getter
    # ====================================
    @property
    def confidenceThreshold(self) -> float:
        return self._confidenceThreshold

    @confidenceThreshold.setter
    def confidenceThreshold(self, conThreshold):
        if not isinstance(conThreshold, float):
            raise TypeError("Confidence Threshold must be of type float.")
        self._confidenceThreshold = conThreshold

    # ==================
    #   Utility Getters
    # ==================
    def detect(self, imagePayload: Image_Payload) -> Inference_Result:
        # Simulate local runtime detection inference pass
        # In deployment, this uses self._confidenceThreshold to filter predictions
        boxes = [Bounding_Box(10.0, 20.0, 50.0, 100.0)]
        return Inference_Result(
            detectedObjects=["vacuum_truck"], 
            confidenceScores=[0.92], 
            boundingBoxes=boxes, 
            timestamp=datetime.now()
        )

    def loadModel(self, version: str) -> bool:
        self._modelVersion = version
        return True

class Truck_Counting_Engine(Image_Process_Engine):
    """Counts trucks and hopefully will catch double counts"""
    def __init__(self, trackingState: Dict[str, Any], temporalWindow: int, processingMode: str, lastProcessedAt: datetime):
        super().__init__(processingMode, lastProcessedAt)
        self.trackingState = trackingState
        self.temporalWindow = temporalWindow

    # ==============================
    #   trackingState getter/setter
    # ==============================
    @property
    def trackingState(self) -> Dict:
        return self._trackingState

    @trackingState.setter
    def trackingState(self, tState):
        if not isinstance(tState, Dict):
            raise TypeError("Tracking State must be of type Dictionary.")
        self._trackingState = tState

    # ===============================
    #   temporalWindow getter/setter
    # ===============================
    @property
    def temporalWindow(self) -> int:
        return self._temporalWindow

    @temporalWindow.setter
    def temporalWindow(self, tWindow):
        if not isinstance(tWindow, int):
            raise TypeError("Temporal Window must be of type int.")
        self._temporalWindow = tWindow

    # ==================
    #   Utility Getters
    # ==================
    def count(self, result: Inference_Result, facilityID: str) -> 'Facility_Metrics_Report':
        # Business logic processes Bounding_Boxes and trackingState over time
        count_value = result.getDetectionCount() 
        summaries = [f"Found {count_value} targets at {result._timestamp}"]
        
        return Facility_Metrics_Report(
            totalTruckCount=count_value,
            reportTimestamp=datetime.now(),
            facilityID=facilityID,
            detectionSummaries=summaries
        )

    def updateTrackingState(self, result: Inference_Result) -> bool:
        return True

    def resetTrackingState(self) -> bool:
        self.trackingState.clear()
        return True

# ===============================
#   3. REPORTING & SYSTEM CORE
# ===============================

class Facility_Metrics_Report:
    """Payload for sending to the UI"""
    def __init__(self, totalTruckCount: int, reportTimestamp: datetime, facilityID: str, detectionSummaries: List[str]):
        self.totalTruckCount = totalTruckCount
        self.reportTimestamp = reportTimestamp
        self.facilityID = facilityID
        self.detectionSummaries = detectionSummaries

    def addDetectionSummary(self, summary: str) -> None:
        self.detectionSummaries.append(summary)

    def getSummaryCount(self) -> int:
        return self.totalTruckCount

    def generateReportText(self) -> str:
        return f"Facility {self.facilityID} Report Summary: Total Count = {self.totalTruckCount}"


class Orchestrator:
    """Main controller of the whole system"""
    def __init__(self, requestCount: int, lastReceivedAt: datetime, detector: Truck_Detection_Model, tracker: Truck_Counting_Engine):
        self.requestCount = requestCount
        self.lastReceivedAt = lastReceivedAt
        self.detector = detector        # Dependency Injection / Association
        self.tracker = tracker          # Dependency Injection / Association

    # =============================
    #   requestCount getter/setter
    # =============================
    @property
    def requestCount(self) -> int:
        return self._requestCount

    @requestCount.setter
    def requestCount(self, reqCount):
        if not isinstance(reqCount, int):
            raise TypeError("Request Count must be of type int.")
        self._requestCount = reqCount

    # ===============================
    #   lastReceivedAt getter/setter
    # ===============================
    @property
    def lastReceivedAt(self) -> datetime:
        return self._lastReceivedAt

    @lastReceivedAt.setter
    def lastReceivedAt(self, lastReceiveAt):
        if not isinstance(lastReceiveAt, datetime):
            raise TypeError("lastReceivedAt must be of type datetime.")
        self._lastReceivedAt = lastReceiveAt

    # ==================
    #   Utility Getters
    # ==================
    def receivePayload(self, rawImage: tuple) -> Image_Payload:
        try:
            return Image_Payload(*rawImage)
        except Exception as e:
            print(f"Error packing imagePayload: {e}")
            return None

    def validatePayload(self, imagePayload: Image_Payload) -> bool:
        if imagePayload and imagePayload.getSize() > 0:
            return True
        return False

    def process_pipeline(self, raw_data_tuple: tuple, facility_id: str) -> Facility_Metrics_Report:
        """Coordinates clean step-by-step pipeline execution matching data requirements."""
        self.requestCount += 1
        self.lastReceivedAt = datetime.now()

        # Step 1: receive raw image
        payload = self.receivePayload(raw_data_tuple)
        if not self.validatePayload(payload):
            raise ValueError("Payload failed system validation criteria.")

        # Step 2: preprocess the imagePayload
        payload = self.detector.preprocess(payload)

        # Step 3: run the model on imagePayload
        inference_out = self.detector.detect(payload)

        # Step 4: track and count trucks
        self.tracker.updateTrackingState(inference_out)
        raw_report = self.tracker.count(inference_out, facility_id)

        # Step 5: postprocess to return a clean Facility Metrics Report
        final_report = self.tracker.postprocess(raw_report)
        return final_report


'''--------------------------------
    RUNTIME VERIFICATION EXAMPLE
--------------------------------'''
if __name__ == "__main__":
    # Create concrete subcomponents 
    vision_model = Truck_Detection_Model("v2.4-obstructed-view", 0.75, "GPU-Batch", datetime.now())
    state_tracker = Truck_Counting_Engine(trackingState={}, temporalWindow=30, processingMode="Stream", lastProcessedAt=datetime.now())

    # Initialize the Orchestrator with its direct processing dependencies
    system_controller = Orchestrator(requestCount=0, lastReceivedAt=datetime.now(), detector=vision_model, tracker=state_tracker)

    # Simulated incoming payload tuple: (imageID, rawBytes, timestamp, metadata)
    dummy_image_data = ("IMG_88291", b'\x89PNG\r\n\x1a\n', datetime.now(), {"facility_zone": "North-Gate"})

    # Run processing loop safely
    generated_report = system_controller.process_pipeline(dummy_image_data, facility_id="SWD-Lubbock-04")
    
    print(f"Successfully populated UI element text: '{generated_report.generateReportText()}'")
    print(f"Total Truck Count: {generated_report.getSummaryCount()}")
