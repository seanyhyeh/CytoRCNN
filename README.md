# CytoRCNN
Extracting global masks from cytorcnn predictions
## Problem with CytoRCNN predictions
CytoRCNN can not immediately produce masks for large images as it requires input images to be 256 by 256. 
Tiling the image into 256 by 256 normally causes problems as one cell could be called in two adjacent tiles and labelled as two distinct cells. In the image shown below of two adjacent images, CytoRCNN produces different calls for the same cells that are in the 256 by 64 overlap (for example we will combine the three circled cells shown.) 
![Screenshot 1](https://github.com/user-attachments/assets/a584c7e1-f34a-4a02-9ff5-ba65e93fada5)\
Instead, we tiled the image as shown below into 256 by 256 with overlaps of 256 by 64 with adjacent tiles.
![Screenshot 2025-05-30 at 7 49 50 PM](https://github.com/user-attachments/assets/94286f13-6008-483e-9829-f089a23a84c7)\
This way, all masks lie entirely within a tile (assuming cell diameter does not exceed 64). If a cell/nuclei call touches the border of the tile, we omit this call as a bordering tile must completely contain the mask. If a cell is contained entirely in the overlap region of 256 by 64, we merge the calls.
## CytoRCNN modifications
I slightly changed the predict function in Cyto-R-CNN/src/cytorcnn/cytorcnn.py in order to return predictions as a python object instead of a json file. 
```
    def predict(self, image): # image_path
        if self.weights_path is None:
            raise RuntimeError("Error: Load a model before calling .predict()")

        # file_paths = [image_path]

        instances_for_each_image = []
        # for path in file_paths:
            # image = cv2.imread(path)
            # result = self.predictor(image)
            # instances = result["instances"]
            # instances_for_each_image.append(instances)
        
        result = self.predictor(image)
        instances = result["instances"]
        instances_for_each_image.append(instances)

        # coco = create_coco_file_from_detectron_instances(
        #     instances_for_each_image, file_paths
        # )
        coco = create_coco_file_from_detectron_instances(
            instances_for_each_image, ["empty"]
        )
        # write_dict_to_file(coco, "prediction.json")
        return(coco)
```
Also, increasing or decreasing `self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST` can increase or decrease number of prediction as the model becomes more loose or strict.
## Method
We start by tiling the given image into tiles of 256 by 256 that overlap 256 by 64 with adjacent tiles. This is done because CytoRCNN requires inputs to be 256 by 256 and the overlap methodology later takes into account overcounting of a mask between two tiles.
![Screenshot 2025-05-30 at 7 53 55 PM](https://github.com/user-attachments/assets/98998b9b-3f1c-40ef-8271-e0d2ee51a2ec)\
We proceed with a by-breadth approach on the diagonal shown in the image above. We start with the tile (0,0), (0,1), (1,0), (0,2), (1,1), (2,0) etc. We store tiles in the current diagonal in the list curr_polygons and we compare overlap with polygons in the previous diagonal (prev_polygons). When we reach a new diagonal, we add polygons stored in prev_polygons to global_polygons, update prev_polygons to be curr_polygons, and set curr_polygons to be an empty list.\\
For each tile, we start by combining asks of the same type (nuclei or cell). Then we combine nuclei that are completely contained in another a cell. This is done using shapely's STRtree. Then, we add these masks into the list curr_polygons.
