vision_model = The AI model we end up using. Most likely some form of YOLO
state_tracker = Object to keep track of our count

system_controller = Main Orchestrator that will handle all the logic in the system

dummy_image_data = Placeholder for actual image input

generated_report = The final report we will send to the UI

NOTE:
  The truck counting logic will have to be adjusted based on the model we use.
  Bounding box overlapping will have to be directly informed by our model
  Image Payload may have to be adjusted based on the structure of the image data. I could've gone ahead and done this but I was more focused
    on the overall structure of the system first.
  Everything else, God willing, should work right out the box
