# CytoRCNN
Extracting global masks from cytorcnn predictions
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
![VMI_test_monocolor](https://github.com/user-attachments/assets/bf249bf4-47d7-4870-8c71-c481845ec6a3)\
\
We start by tiling the given image into tiles of 256 by 256 that overlap 256 by 64 with adjacent tiles. This is done because CytoRCNN requires inputs to be 256 by 256 and the overlap methodology later takes into account overcounting of a mask between two tiles.
\
![VMI_test_diagonalcolor](https://github.com/user-attachments/assets/48e4f8fb-7ac0-47d6-bc4d-66701797bb0e)\
\
We proceed with a by-breadth approach on the diagonal shown in the image above. We start with the tile (0,0), (0,1), (1,0), (0,2), (1,1), (2,0) etc. We store tiles in the current diagonal in the list curr_polygons and we compare overlap with polygons in the previous diagonal (prev_polygons). When we reach a new diagonal, we add polygons stored in prev_polygons to global_polygons, update prev_polygons to be curr_polygons, and set curr_polygons to be an empty list.\\
For each tile, we start by combining asks of the same type (nuclei or cell). Then we combine nuclei that are completely contained in another a cell. This is done using shapely's STRtree. Then, we add these masks into the list curr_polygons.
