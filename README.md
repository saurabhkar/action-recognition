# action-recognition

<p>
The major final year project.ipynb extract data from HMDB51 dataset and process it using various algorithms.
</p>
<p> recognition.py is the unchanged code and no modification has been done on that. </p>
<p>
<a href ="https://gluon-cv.mxnet.io/build/examples_datasets/hmdb51.html"  target="_blank"> link to the complete dataset can be downloaded from here. CLICK HERE .  </a>
</p>

<p><ul>
<li> Some random images were taken from each class ( total number of classes =51) and datafame was made using those images. </li>
<li> SIFT algorithm was applied on that dataframe to extract the SIF points (keypoints and descriptors).</li>

<li> HOG was applied on the original dataframe to extract HOG features. </li>
<li> Both the above algorithms features have been classified using SVM and DNN. </li>
<li> SIFT and HOG feature combination has also been implemented and classified using SVM and DNN. </li> 
<li> CNN was used independently directly on the original dataframe. </li>



</ul>
