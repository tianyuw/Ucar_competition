clear all; close all;
%%
%
% Please modify the code end with %%%%%
% Author: tianyu wang
% Creation(11/13/2016) : Convert the ucar competition dataset to MSCNN
% standard dataset(window file)
% 
%%

root_dir = '/home/tianyuw/DNN/DATA/competition_data/'; %%%% competition data root path
image_dir = [root_dir 'training/training/']; %%%% image files path
label_dir = [root_dir 'training/training/label.txt']; %%%%label file path
interested_object = '1'; %%%%1 -> vehicle 2 -> pedestrain 3 -> cyclist 20-> traffic light
show = 0; %%%% whether display the boundingbox result

image_files = dir(strcat(image_dir, '*.jpg'));
if(strcmp(interested_object, '1'))
    file_name = sprintf('/home/tianyuw/DNN/mscnn/data/kitti/window_files_ucar/mscnn_window_file_ucar_vehicle_train.txt'); %%%%output path
    fid = fopen(file_name, 'wt');
elseif(strcmp(interested_object, '2'))
    file_name = sprintf('/home/tianyuw/DNN/mscnn/data/kitti/window_files_ucar/mscnn_window_file_ucar_pedestrain_train.txt');%%%%output path
    fid = fopen(file_name, 'wt');
elseif(strcmp(interested_object, '3'))
    file_name = sprintf('/home/tianyuw/DNN/mscnn/data/kitti/window_files_ucar/mscnn_window_file_ucar_cyclist_train.txt');%%%%output path
    fid = fopen(file_name, 'wt');
elseif(strcmp(interested_object, '20'))
    file_name = sprintf('/home/tianyuw/DNN/mscnn/data/kitti/window_files_ucar/mscnn_window_file_ucar_traffic_light_train.txt');%%%%output path
    fid = fopen(file_name, 'wt');
end


if (show)
  fig = figure(1); set(fig,'Position',[-30 30 960 300]);
  hd.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

flabel = fopen(label_dir, 'r');
object_total_num = 0;
for i = 1 : size(image_files, 1)
    if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, size(image_files, 1)); end
    img_path = strcat(image_dir, image_files(i, :).name);
    I = imread(img_path);
    if (show)
        imshow(I); axis(hd.axes,'image','off'); hold(hd.axes, 'on');
    end
    [imgH, imgW, channels] = size(I);
    
    fprintf(fid, '# %d\n', i-1);
    fprintf(fid, '%s\n', img_path);
    fprintf(fid, '%d\n%d\n%d\n', channels, imgH, imgW);
  
    tline=fgetl(flabel);
    if (isempty(strfind(tline, image_files(i, :).name)))
        fprintf('image file name and the name in label file doesnt match');
        return;
    end
    
    %% parse each line of label file by finding '[[', ']]', '], ['
    label_start = strfind(tline, '[[');
    label_end = strfind(tline, ']]');
    
    label_string = tline(label_start + 2: label_end - 1);
    label_split = strfind(label_string, '], [');
    label_num = length(label_split) + 1;
    sub_string_loc = [1 label_split length(label_string)];
    object_num = 0;
    sub_label_string = [];
    for j = 1 : label_num
        if (length(label_string) == 0)
            object_num = 0;
        elseif(j == 1 && j~=label_num)
            sub_label_string = label_string(sub_string_loc(j) : (sub_string_loc(j + 1)) - 1);
        elseif(j == label_num && j~=1)
            sub_label_string = label_string((sub_string_loc(j) + 4) : sub_string_loc(j + 1));
        elseif(j == label_num && j == 1)
            sub_label_string = label_string(sub_string_loc(j) : end);
        else
            sub_label_string = label_string((sub_string_loc(j) + 4) : (sub_string_loc(j + 1)) - 1);
        end
        % find the number of vehicle object
        if (~isempty(sub_label_string))
            if (strcmp(sub_label_string(end), interested_object))
                object_num = object_num + 1;   
            end
        end
    end
    fprintf(fid, '%d\n', object_num);
    object_total_num = object_total_num + object_num;
    for j = 1 : label_num
        if(object_num == 0)
            continue;
        elseif(j == 1 && j~=label_num)
            sub_label_string = label_string(sub_string_loc(j) : (sub_string_loc(j + 1)) - 1);
        elseif(j == label_num && j~=1)
            sub_label_string = label_string((sub_string_loc(j) + 4) : sub_string_loc(j + 1));
        elseif(j == label_num && j == 1)
            sub_label_string = label_string(sub_string_loc(j) : end);
        else
            sub_label_string = label_string((sub_string_loc(j) + 4) : (sub_string_loc(j + 1)) - 1);
        end
        if (~isempty(sub_label_string))
            if (strcmp(sub_label_string(end), interested_object))
                comma_loc = strfind(sub_label_string, ',');
                x1 = str2double(sub_label_string(1 : comma_loc(1) - 1));
                y1 = str2double(sub_label_string((comma_loc(1) + 2) : (comma_loc(2) - 1)));
                x2 = str2double(sub_label_string((comma_loc(2) + 2) : (comma_loc(3) - 1)));
                y2 = str2double(sub_label_string((comma_loc(3) + 2) : (comma_loc(4) - 1)));
                label = str2double(sub_label_string((comma_loc(4) + 2) : end));
                w = x2 - x1 + 1; h = y2 - y1 + 1;
                fprintf(fid, '%d %d %d %d %d %d\n', label, 0, round(x1), round(y1), round(x2), round(y2));
                if (show)
                    color = 'r';
                    rectangle('Position', [x1, y1, w, h], 'LineWidth', 2, 'edgecolor', color);
                    text(x1+0.5*w, y1, interested_object, 'color', 'r', 'BackgroundColor', 'k', 'HorizontalAlignment',...
                    'center','VerticalAlignment','bottom','FontWeight','bold','FontSize',8);
                end
            end
        end
    end
    fprintf(fid, '0\n'); % 0 means no region-of-non-interested
    if (show), pause(1); end
end
fprintf('window file generation done! total objects = %d\n', object_total_num);
fclose(fid);
