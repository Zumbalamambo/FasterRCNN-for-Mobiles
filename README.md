# Sử dụng Faster R-CNN, Mask R-CNN và SSD nhận diện số thẻ ATM
![](assets/detection_masks.png)


Presentation gồm:
* Kiến trúc mạng Faster R-CNN và demo nhận dạng số thẻ ATM trên Jupyter Notebook
* Kiến trúc mạng Mask R-CNN và demo nhận dạng số thẻ ATM trên Jupyter Notebook
* So sánh Faster R-CNN và Mask R-CNN
* Sử dụng mạng nơ ron trên mobile devices: SSD (Single Shot Detector)
* Demo nhận dạng số thẻ ATM trên điện thoại Android 
* Link tài liệu nghiên cứu

## 1. Kiến trúc mạng Faster R-CNN và demo nhận dạng số thẻ ATM trên Jupyter Notebook
Visualizes every step of the first stage Region Proposal Network and displays positive and negative anchors along with anchor box refinement.
![](assets/detection_anchors.png)

## 2. Kiến trúc mạng Mask R-CNN và demo nhận dạng số thẻ ATM trên Jupyter Notebook
This is an example of final detection boxes (dotted lines) and the refinement applied to them (solid lines) in the second stage.
![](assets/detection_refinement.png)

## 3. So sánh Faster R-CNN và Mask R-CNN
Examples of generated masks. These then get scaled and placed on the image in the right location.

![](assets/detection_masks.png)

## 4. Sử dụng mạng nơ ron trên mobile devices: SSD (Single Shot Detector)
Often it's useful to inspect the activations at different layers to look for signs of trouble (all zeros or random noise).

![](assets/detection_activations.png)

## 5. Demo nhận dạng số thẻ ATM trên điện thoại Android
Another useful debugging tool is to inspect the weight histograms. These are included in the inspect_weights.ipynb notebook.

![](assets/detection_histograms.png)

## 6. Link tài liệu nghiên cứu
TensorBoard is another great debugging and visualization tool. The model is configured to log losses and save weights at the end of every epoch.

