%% ---------conv2~conv5权重可视化函数----------------
function []= visualize_weights(w, s)
h = max(size(w, 1), size(w, 2));             % kernel size
g = h + s;                     % grid size, larger than kernel size for better visual effects.

% Normalization for gray scale
w = w - min(min(min(min(w))));
w = w / max(max(max(max(w)))) * 255;
w = uint8;

W = zeros(g*size(w,3), g*size(w,4));
for u = 1:size(w,3)
    for v = 1:size(w,4)
        W(g*(u-1) + (1:h), g*(v-1) + (1:h)) = w(:,:,u,v)';
    end
end
W = uint8(W);
figure;
imshow(W);
end


