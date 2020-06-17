%% -----------mainº¯Êý£¬caffenet_weights_vis.m -------------------------
clear;
clc;
close all;
addpath('matlab')
caffe.set_mode_cpu();
fprintf(['caffe version = ', caffe.version(), '\n']);
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt' ... 
                ,'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 'test');
fprintf('Load net done. Net layers: ');
net.layer_names
fprintf('Net blobs: ');
net.blob_names

%conv1 weight visualization
conv1_layer = net.layer_vec(2);
blob1 = conv1_layer.params(1);
w = blob1.get_data();
fprintf('conv1 weight shape: ');
size(w)
visualize_weights(w,1);

%conv2 weight visualization
conv2_layer = net.layer_vec(6);
bolb2 = conv2_layer.params(1);
w2 = blob2.get_data();
fprintf('conv2 weight shape: ');
size(w2)
visualize_weights(w2,1);

%conv3 weight visualization
conv3_layer = net.layer_vec(10);
blob3 = conv3_layer.params(1);
w3 = blob3.get_data();
fprintf('conv3 weight shape: ');
size(w3)
visualize_weights(w3,1);

%conv4 weight visualization
conv4_layer = net.layer_vec(12);
blob4 = conv4_layer.params(1);
w4 = blob4.get_data();
fprintf('conv4 weight shape: ');
size(w4)
visualize_weights(w4,1);

%conv5 weight visualization
conv5_layer = net.layer_vec(14);
blob5 = conv5_layer.params(1);
w5 = blob5.get_data();
fprintf('conv5 weight shape: ');
size(w5)
visualize_weights(w5,1);
