% Simple webcam demo. Train the CNN first to create trainedDrowsinessNet.mat.
clear; clc; close all;
if ~isfile("trainedDrowsinessNet.mat")
    error("Run train_drowsiness_cnn.m first.");
end
S = load("trainedDrowsinessNet.mat", "trainedDrowsinessNet");
net = S.trainedDrowsinessNet;
alertThreshold = 15;
closedCount = 0;

cams = webcamlist;
if isempty(cams), error("No webcam detected or MATLAB webcam support unavailable."); end
cam = webcam(1);

detectorAvailable = false;
try
    eyeDetector = vision.CascadeObjectDetector("EyePairBig");
    detectorAvailable = true;
catch
    warning("EyePairBig detector unavailable; using the full frame.");
end

fig = figure("Name", "Drowsiness Detection", "NumberTitle", "off");
while ishandle(fig)
    frame = snapshot(cam);
    roi = frame;
    if detectorAvailable
        boxes = eyeDetector(frame);
        if ~isempty(boxes)
            [~, ix] = max(boxes(:,3).*boxes(:,4));
            roi = imcrop(frame, boxes(ix,:));
        end
    end
    roi = imresize(roi, [64 64]);
    if size(roi,3) == 1, roi = repmat(roi,1,1,3); end
    if size(roi,3) > 3, roi = roi(:,:,1:3); end

    predicted = string(classify(net, roi));
    if contains(lower(predicted), "closed")
        closedCount = closedCount + 1;
    else
        closedCount = 0;
    end

    imshow(frame);
    if closedCount >= alertThreshold
        title("DROWSINESS ALERT - prolonged closed-eye prediction", "Color", "r");
    else
        title("Predicted eye state: " + predicted);
    end
    drawnow;
end
clear cam;
