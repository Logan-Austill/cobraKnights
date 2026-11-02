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
        """Returns the area of the bounding box"""
        return self.width * self.height

    # checking to see if multiple bounding boxes are overlapped
    def intersection(self, other: 'Bounding_Box') -> bool:
        """Returns boolean for whether or not bounding boxes are overlapping"""
        # Placeholder for overlapping geometric calculation
        # Depends on how strict we want an intersection to be
        return True

    # should help us distinguish boxes if they are overlapped
    def center(self) -> tuple:
        """Returns the coordinates of the center of the box"""
        return (self.x + self.width / 2, self.y + self.height / 2)


class Image_Payload:
    """Packages a raw image with information we need for processing"""
    def __init__(self, imageID: str, rawBytes: bytes, timestamp: datetime, metadata: Dict[str, Any]):
        self.imageID = imageID
        self.rawBytes = rawBytes
        self.timestamp = timestamp
        self.metadata = metadata

    def getMetaData(self, key: str) -> Any:
        """Returns meta data of image"""
        return self.metadata.get(key)

    def updateMetadata(self, key: str, value: Any) -> bool:
        """Returns boolean on a successful meta data update"""
        self.metadata[key] = value
        return True

    def getSize(self) -> int:
        """Returns size of image"""
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
        """Returns detectedObjects list"""
        return self._detectedObjects

    @detectedObjects.setter
    def detectedObjects(self, detObjList):
        """Setter checks if detectedObjects is in list form"""
        if not isinstance(detObjList, list):
            raise TypeError("Detected Objects must be of type list[str].")
        self._detectedObjects = detObjList   

    # =================================
    #   confidenceScores getter/setter
    # =================================
    @property
    def confidenceScores(self) -> List[float]:
        """Returns confidenceScores list"""
        return self._confidenceScores
    
    @confidenceScores.setter
    def confidenceScores(self, confScoreList):
        """Setter checks if confidenceScores is in list form"""
        if not isinstance(confScoreList, list):
            raise TypeError("Confidence Scores must be of type list[float].")
        self._confidenceScores = confScoreList

    # ==============================
    #   boundingBoxes getter/setter
    # ==============================
    @property
    def boundingBoxes(self) -> List[Bounding_Box]:
        """Returns list of bounding boxes"""
        return self._boundingBoxes
    
    @boundingBoxes.setter
    def boundingBoxes(self, boundBoxList):
        """Setter checks if boundingBoxes is in list form"""
        if not isinstance(boundBoxList, list):
            raise TypeError("Bounding Boxes must be of type list[Bounding_Box].")
        self._boundingBoxes = boundBoxList

    # ==========================
    #   timestamp getter/setter
    # ==========================
    @property
    def timestamp(self) -> datetime:
        """Returns timestamp"""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, dtStamp):
        """Setter checks if timestamp is in datetime form"""
        if not isinstance(dtStamp, datetime):
            raise TypeError("Timestamp must be of type datetime.")
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
        """Returns processingMode"""
        return self._processingMode

    @processingMode.setter
    def processingMode(self, pMode):
        """Checks to make sure processingMode input is a string"""
        if not isinstance(pMode, str):
            raise TypeError("Processing Mode must be of type string.")
        self._processingMode = pMode

    # ================================
    #   lastProcessedAt getter/setter
    # ================================
    @property
    def lastProcessedAt(self) -> datetime:
        """Returns lastProcessedAt timestamp"""
        return self._lastProcessedAt

    @lastProcessedAt.setter
    def lastProcessedAt(self, lastPTime):
        """Checks if lastProcessedAt is datetime"""
        if not isinstance(lastPTime, datetime):
            raise TypeError("lastProcessedAt must be of type datetime.")
        self._lastProcessedAt = lastPTime

    # ==================
    #   Utility Getters
    # ==================
    def preprocess(self, imagePayload: Image_Payload) -> Image_Payload:
        """Returns imagePayload"""
        return imagePayload

    def postprocess(self, report: 'Facility_Metrics_Report') -> 'Facility_Metrics_Report':
        """Returns FMR"""
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
        """Returns mdoelVersion"""
        return self._modelVersion

    @modelVersion.setter
    def modelVersion(self, mVersion):
        """Checks if modelVersion is a string"""
        if not isinstance(mVersion, str):
            raise TypeError("Model Version must be of type string.")
        self._modelVersion = mVersion
    
    # ====================================
    #   confidenceThreshold setter/getter
    # ====================================
    @property
    def confidenceThreshold(self) -> float:
        """Returns confidenceThreshold"""
        return self._confidenceThreshold

    @confidenceThreshold.setter
    def confidenceThreshold(self, conThreshold):
        """Checks if confidenceThreshold is a float"""
        if not isinstance(conThreshold, float):
            raise TypeError("Confidence Threshold must be of type float.")
        self._confidenceThreshold = conThreshold

    # ==================
    #   Utility Getters
    # ==================
    def detect(self, imagePayload: Image_Payload) -> Inference_Result:
        """Detects objects and returns Inference_Result"""
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
        """Returns boolean representing if model was loaded correctly"""
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
        """Returns trackingState"""
        return self._trackingState

    @trackingState.setter
    def trackingState(self, tState):
        """Checks if trackingState is a Dict"""
        if not isinstance(tState, Dict):
            raise TypeError("Tracking State must be of type Dictionary.")
        self._trackingState = tState

    # ===============================
    #   temporalWindow getter/setter
    # ===============================
    @property
    def temporalWindow(self) -> int:
        """Returns temporalWindow"""
        return self._temporalWindow

    @temporalWindow.setter
    def temporalWindow(self, tWindow):
        """Checks if temporalWindow is an integer"""
        if not isinstance(tWindow, int):
            raise TypeError("Temporal Window must be of type int.")
        self._temporalWindow = tWindow

    # ==================
    #   Utility Getters
    # ==================
    def count(self, result: Inference_Result, facilityID: str) -> 'Facility_Metrics_Report':
        """Counts trucks and returns FMR"""
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
        """Returns boolean representing TrackingState update success"""
        return True

    def resetTrackingState(self) -> bool:
        """Returns boolean representing TrackingState reset success"""
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
        """Appends summary to detectionSummaries list"""
        self.detectionSummaries.append(summary)

    def getSummaryCount(self) -> int:
        """Returns totalTruckCount"""
        return self.totalTruckCount

    def generateReportText(self) -> str:
        """Returns string representation of FMR"""
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
        """Returns requestCount"""
        return self._requestCount

    @requestCount.setter
    def requestCount(self, reqCount):
        """Checks if requestCount is an integer"""
        if not isinstance(reqCount, int):
            raise TypeError("Request Count must be of type int.")
        self._requestCount = reqCount

    # ===============================
    #   lastReceivedAt getter/setter
    # ===============================
    @property
    def lastReceivedAt(self) -> datetime:
        """Returns lastReceivedAt timestamp"""
        return self._lastReceivedAt

    @lastReceivedAt.setter
    def lastReceivedAt(self, lastReceiveAt):
        """Checks if lastReceivedAt is in datetime form"""
        if not isinstance(lastReceiveAt, datetime):
            raise TypeError("lastReceivedAt must be of type datetime.")
        self._lastReceivedAt = lastReceiveAt

    # ==================
    #   Utility Getters
    # ==================
    def receivePayload(self, rawImage: tuple) -> Image_Payload:
        """Returns Image_Payload form of rawImage"""
        try:
            return Image_Payload(*rawImage)
        except Exception as e:
            print(f"Error packing imagePayload: {e}")
            return None

    def validatePayload(self, imagePayload: Image_Payload) -> bool:
        """Returns boolean representing imagePayload is of type imagePayload and not empty"""
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
