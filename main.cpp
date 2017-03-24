/*
 * main.cpp
 *
 *  Created on: Dec 13, 2016
 *      Author: tianyuw
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "gpu_nms.hpp"
#include <time.h>
#include "caffe/caffe.hpp"
#include <dirent.h>
#include <math.h>
#include <sstream>


//#define MSCNN_VEHICLE 1


using namespace std;
using namespace cv;
using namespace caffe;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

//#if MSCNN_VEHICLE
/*
const int class_num = 1;
#else
const int class_num = 2;
#endif
*/

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  cpuSecond
 *  Description:  Timing with CPU timer
 * =====================================================================================
 */
double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/*
 * ===  Class  ======================================================================
 *         Name:  Show_Detection
 *  Description:  Show the MSCNN detection result
 * =====================================================================================
 */
class Show_Detection{
public:
	void add_bbox(vector<float> & add_box){bboxes.push_back(add_box);};
	vector<float> get_bbox(int loc){return bboxes[loc];};
	int bbox_size(){
		return bboxes.size();
	};
	void clear_bboxes(){bboxes.clear();};
	void show_bboxes();
	Mat disp_image; //pre-store the test image just for draw bbox
	enum bbox_field{X1, Y1, X2, Y2, LABEL, CONFIDENCE, NUM};
protected:

	static vector<vector<float> > bboxes; // car has label 1, ped has label 2, cyc has label 3, TL has label 4
};
vector<vector<float> > Show_Detection::bboxes(0);

void Show_Detection::show_bboxes(){
	int bboxes_num = bboxes.size();
	float x1, y1, x2, y2, label, confidence;
	char text_msg[20];
	cv::Size text_size;

	// show description on image
	cv::putText(disp_image, "Vehicle: Red", cv::Point(25, 25), FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255, 0));
	cv::putText(disp_image, "Pedestrian: Blue", cv::Point(150, 25), FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 0, 0, 0));
	cv::putText(disp_image, "Cyclist: Green", cv::Point(320, 25), FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0, 0));
	cv::putText(disp_image, "Traffic Light: Orange", cv::Point(450, 25), FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 127, 255, 0));

	for(vector<vector<float> > :: iterator it = bboxes.begin(); it != bboxes.end(); ++it)
	{
		x1 = (*it)[X1];
		y1 = (*it)[Y1];
		x2 = (*it)[X2];
		y2 = (*it)[Y2];
		label = (*it)[LABEL];
		confidence = (*it)[CONFIDENCE];

		if(label == 1)
			cv::rectangle(disp_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255, 0), 1.5);
		else if(label == 2)
			cv::rectangle(disp_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 0), 1.5);
		else if (label == 3)
			cv::rectangle(disp_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0, 0), 1.5);
		else if (label == 20)
			cv::rectangle(disp_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 127, 255, 0), 1.5);


		sprintf(text_msg, "%2.2f", confidence);
		text_size = getTextSize(text_msg, FONT_HERSHEY_DUPLEX, 0.4, 1, NULL);
		//cout << text_size << endl;
		cv::putText(disp_image, text_msg, cv::Point((x1 + x2) / 2 - text_size.width / 2, y1 - 5), FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0, 0, 255, 0));
		//ss.str("");
	}



}

/*
 * ===  Class  ======================================================================
 *         Name:  Write to Json
 *  Description:  Output the detected bbox to Json file
 * =====================================================================================
 */
class Write_Json:public Show_Detection{ //inorder to use variable bbox
public:
	Write_Json(string & json_path){
		json_file.open(json_path.c_str(), ios::out);
	    if (!json_file.is_open()){
	    	cout << "json file can not be opened!" << endl;
	    }
	    json_file << "{\n" ;
	}
	~Write_Json(){
		json_file << "}\n";
		json_file.close();
	}
	ofstream json_file;
	void writeToJson(string & image_name, int);

};

void Write_Json::writeToJson(string & image_name, int remain_img){
	float x1, y1, x2, y2, label, confidence;

	string one_line = "\"" + image_name + "\":[";
	stringstream one_bbox;

	for(vector<vector<float> > :: iterator it = bboxes.begin(); it != bboxes.end(); ++it)
	{
			x1 = (*it)[X1];
			y1 = (*it)[Y1];
			x2 = (*it)[X2];
			y2 = (*it)[Y2];
			label = (*it)[LABEL];
			confidence = (*it)[CONFIDENCE];
			if (it != bboxes.end() - 1)
				one_bbox << "[" << x1 <<", " << y1 << ", " << x2 << ", " << y2 << ", " << (int)label << ", " << confidence << "], ";
			else
				one_bbox << "[" << x1 <<", " << y1 << ", " << x2 << ", " << y2 << ", " << (int)label << ", " << confidence << "]";
			one_line += one_bbox.str();
			one_bbox.str("");
	}
	if(remain_img != 1)
		one_line += "],\n";
	else
		one_line += "]\n";
	json_file << one_line;

}





/*
 * ===  Class  ======================================================================
 *         Name:  Detector_mscnn
 *  Description:  MSCNN Detector
 * =====================================================================================
 */
class Detector_mscnn {
public:
	Detector_mscnn(const string& model_file, const string& weights_file, int W, int H, bool detect_flag, int num_of_class, int label_of_class);
    virtual void Detection(Mat & cv_img, Show_Detection & disp_img, float PROP_THRESH);
    virtual void bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, float ratio_W, float ratio_H, int img_width, int img_height);
    void vis_detections(cv::Mat & image, int* keep, int & num_out, float* sorted_pred_cls, float & CONF_THRESH);
    //void vis_detections(cv::Mat image, int num, float* sorted_pred_cls, float CONF_THRESH);
    void boxes_sort(int num, const float* pred, float* sorted_pred);
    enum Object{VEHICLE, PED_CYC, CYC, TRAFFIC_LIGHT};
private:
    shared_ptr<Net<float> > net_;
    int image_W;
    int image_H;
    bool car_or_not;
    int class_num;
    int class_label;
};

class Detector_rfcn : public Detector_mscnn{
public:
	//Detector_rfcn(const string& model_file, const string& weights_file) : Detector_mscnn(model_file, weights_file){};
};

struct Info
{
    float score;
    const float* head;
};



Detector_mscnn::Detector_mscnn(const string& model_file, const string& weights_file, int W, int H, bool detect_flag, int num_of_class, int label_of_class)
{
    net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    image_W = W;
    image_H = H;
    car_or_not = detect_flag;
    class_num = num_of_class;
    class_label = label_of_class;
}

void Detector_mscnn::Detection(Mat & cv_img, Show_Detection & result, float PROP_THRESH)
{
	float CONF_THRESH = 0.6;
	float NMS_THRESH = 0.5;
	int width = cv_img.cols;
	int height = cv_img.rows;

	float ratio_H = (float) image_H / (float) height;
	float ratio_W = (float) image_W / (float) width;

	int num_out;
	Mat cv_resized_int(image_H, image_W, CV_8UC3, cv::Scalar(0, 0, 0));
	//Mat cv_resized(image_H, image_W, CV_32FC3, cv::Scalar(0, 0, 0));
	//std::cout<<"imagename "<<im_name<<endl;
	resize(cv_img, cv_resized_int, Size(image_W, image_H));

	float data_buf[image_H * image_W * 3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt;
	float * bbox_delt_prop = NULL;
	const float* rois;
	const float* pred_cls;
	float * pred_cls_prop = NULL;
	int num, num_prop;

	for (int h = 0; h < image_H; ++h) {
		for (int w = 0; w < image_W; ++w) {
			data_buf[(0 * image_H + h) * image_W + w] = float(
					cv_resized_int.at<cv::Vec3b>(cv::Point(w, h))[0])
					- float(104.0);
			data_buf[(1 * image_H + h) * image_W + w] = float(
					cv_resized_int.at<cv::Vec3b>(cv::Point(w, h))[1])
					- float(117.0);
			data_buf[(2 * image_H + h) * image_W + w] = float(
					cv_resized_int.at<cv::Vec3b>(cv::Point(w, h))[2])
					- float(123.0);
		}
	}

	net_->blob_by_name("data")->Reshape(1, 3, image_H, image_W);
	net_->blob_by_name("data")->set_cpu_data(data_buf);
	//net_->blob_by_name("im_info")->set_cpu_data(im_info);
	net_->ForwardFrom(0);
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	pred_cls = net_->blob_by_name("cls_pred")->cpu_data();
	rois = net_->blob_by_name("proposals_score")->cpu_data();
	num = net_->blob_by_name("proposals_score")->num();

	boxes = new float[num * 4];
	bbox_delt_prop = new float[num * 20];
	pred_cls_prop = new float[num * 5];
	pred = new float[num * 5 * class_num];
	pred_per_class = new float[num * 5];
	sorted_pred_cls = new float[num * 5];
	keep = new int[num];
	//***************************************check data*****************************************
	/*    cout << "roi[0]: " << endl;
	 cout << rois[0] << " " << rois[1] << " " << rois[2] << " " << rois[3] << " " << rois[4] << " "
	 << rois[5] << " " <<rois[6] << " " <<rois[7] << " " <<rois[8] << " " <<rois[9] << " " <<rois[10] << endl;*/
	//******************************************************************************************
/*	for (int n = 0; n < num; n++) {
		for (int c = 0; c < 4; c++) {
			//if (c == 0 || c == 2){
			boxes[n * 4 + c] = rois[n * 6 + c + 1];

		}
	}*/

    num_prop = 0;
    int n = 0, cols_num;
    while (n < num){
    	if (rois[n * 6 + 5] > PROP_THRESH){
    		for (int c = 0; c < 4; c++){
    			boxes[num_prop*4+c] = rois[n*6+c+1];
    		}
    		if(car_or_not){
    			cols_num = 20;
    		}else{
    			cols_num = 4 * (class_num + 1);
    		}
    		for (int c = 0; c < cols_num; c++)
    		{
    			bbox_delt_prop[num_prop * cols_num + c] = bbox_delt[n * cols_num + c];
    		}
    		if(car_or_not){
    			cols_num = 5;
    		}else{
    			cols_num = class_num + 1;
    		}
    		for (int c = 0; c < cols_num; c++)
    		{
    			pred_cls_prop[num_prop * cols_num + c] = pred_cls[n * cols_num + c];
    		}
			num_prop++;
			n++;
    	}else{
    		n++;
    		continue;
    	}
    }

	bbox_transform_inv(num_prop, bbox_delt_prop, pred_cls_prop, boxes, pred, ratio_W, ratio_H,
			width, height);

	//*************************************show 'pred' for final bbox******************************
	/*    cout << "pred: " << endl;
	 cout << pred[0] << ", " << pred[1] << ", " << pred[2] << ", " << pred[3] << ", " << pred[4]<< endl;
	 rectangle(cv_img, Point(pred[0], pred[1]), Point(pred[2], pred[3]), Scalar(0, 0, 255, 0), 2.0, 8, 0);
	 imshow("test", cv_img);
	 waitKey(0);*/
	//*********************************************************************************************
	vector<float> bbox(Show_Detection::NUM); //the space of the container is defined, so can use assign operator
	if (car_or_not) {
		boxes_sort(num_prop, pred, sorted_pred_cls);
		if (num_prop != 0) {
			_nms(keep, &num_out, sorted_pred_cls, num_prop, 5, NMS_THRESH, 0);
			for (int coor = 0; coor < num_out; coor++) {
/*				cout << sorted_pred_cls[keep[coor] * 5 + 0] << ", "
						<< sorted_pred_cls[keep[coor] * 5 + 1] << ", "
						<< sorted_pred_cls[keep[coor] * 5 + 2] << ", "
						<< sorted_pred_cls[keep[coor] * 5 + 3] << endl;*/
				if(sorted_pred_cls[keep[coor] * 5 + 4]>CONF_THRESH){
					bbox[Show_Detection::X1] = sorted_pred_cls[keep[coor] * 5 + 0];
					bbox[Show_Detection::Y1] = sorted_pred_cls[keep[coor] * 5 + 1];
					bbox[Show_Detection::X2] = sorted_pred_cls[keep[coor] * 5 + 2];
					bbox[Show_Detection::Y2] = sorted_pred_cls[keep[coor] * 5 + 3];
					bbox[Show_Detection::LABEL] = class_label + 1;
					bbox[Show_Detection::CONFIDENCE] = sorted_pred_cls[keep[coor] * 5 + 4];

					result.add_bbox(bbox);
				}
			}
/*			vis_detections(disp_img, keep, num_out, sorted_pred_cls,
					CONF_THRESH);*/

		}
	} else {
		for (int i = 0; i < class_num; i++) {

			for (int j = 0; j < num_prop; j++) {
				for (int k = 0; k < 5; k++) {
					pred_per_class[j * 5 + k] =
							pred[(i * num_prop + j) * 5 + k];
					//cout << pred[(i * num_prop + j) * 5 + k] <<endl;
				}
			}
			boxes_sort(num_prop, pred_per_class, sorted_pred_cls);
			if (num_prop != 0) {
				_nms(keep, &num_out, sorted_pred_cls, num_prop, 5, NMS_THRESH,
						0);
				for (int coor = 0; coor < num_out; coor++) {
/*					cout << sorted_pred_cls[keep[coor] * 5 + 0] << ", "
							<< sorted_pred_cls[keep[coor] * 5 + 1] << ", "
							<< sorted_pred_cls[keep[coor] * 5 + 2] << ", "
							<< sorted_pred_cls[keep[coor] * 5 + 3] << endl;*/
					if (sorted_pred_cls[keep[coor] * 5 + 4] > CONF_THRESH) {
						bbox[Show_Detection::X1] = sorted_pred_cls[keep[coor]
								* 5 + 0];
						bbox[Show_Detection::Y1] = sorted_pred_cls[keep[coor]
								* 5 + 1];
						bbox[Show_Detection::X2] = sorted_pred_cls[keep[coor]
								* 5 + 2];
						bbox[Show_Detection::Y2] = sorted_pred_cls[keep[coor]
								* 5 + 3];
						if (class_label == Detector_mscnn::PED_CYC)
							bbox[Show_Detection::LABEL] = class_label + i + 1;
						else if(class_label == Detector_mscnn::TRAFFIC_LIGHT)
							bbox[Show_Detection::LABEL] = 20;
						bbox[Show_Detection::CONFIDENCE] =
								sorted_pred_cls[keep[coor] * 5 + 4];

						result.add_bbox(bbox);
					}
				}

/*				vis_detections(disp_img, keep, num_out, sorted_pred_cls,
						CONF_THRESH);*/
/*				vis_detections(disp_img, num_prop, sorted_pred_cls,
										CONF_THRESH);*/
			}
		}
	}


	delete[] boxes;
	delete[] bbox_delt_prop;
	delete[] pred_cls_prop;
	delete[] pred;
	delete[] pred_per_class;
	delete[] keep;
	delete[] sorted_pred_cls;
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
void Detector_mscnn::bbox_transform_inv(int num, const float* box_deltas,
		const float* pred_cls, float* boxes, float* pred, float ratio_W,
		float ratio_H, int img_width, int img_height) {
	//float * box_deltas
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y,
			pred_w, pred_h, pred_x1, pred_y1, pred_x2, pred_y2;
	float exp_score, sum_exp_score;

	for (int i = 0; i < num; i++) {
		width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1.0;
		height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1.0;
		ctr_x = boxes[i * 4 + 0] + 0.5 * width;
		ctr_y = boxes[i * 4 + 1] + 0.5 * height;

		/******************************************show 'box_deltas' and 'pred_cls'***********************/
		/*        cout << endl;
		 cout<< "box_deltas(0~10): " << endl;
		 cout << box_deltas[0] << ", " << box_deltas[1] << ", " << box_deltas[2] << ", " << box_deltas[3] << ", " << box_deltas[4] << ", " << box_deltas[5] << ", " <<
		 box_deltas[6] << ", " << box_deltas[7] << ", " << box_deltas[8] << ", " << box_deltas[9] << ", " << box_deltas[10]  << endl;
		 cout << endl;
		 cout << "box_deltas(20~30): " << endl;
		 for (int debug = 20; debug < 30; debug ++)
		 {
		 cout << box_deltas[debug] << ", ";
		 }
		 cout << endl;

		 cout << "pred_cls(0~10): " << endl;
		 for (int debug = 0; debug < 10; debug ++)
		 {
		 cout << pred_cls[debug] << ", ";
		 }
		 cout << endl;*/
		/*************************************************************************************************/

		if (car_or_not) {
			dx = box_deltas[(i * 20) + 4];
			dy = box_deltas[(i * 20) + 5];
			dw = box_deltas[(i * 20) + 6];
			dh = box_deltas[(i * 20) + 7];

			// bbox de-normalization
			dx *= 0.10f;
			dy *= 0.10f;
			dw *= 0.20f;
			dh *= 0.20f;

			/********************************************show dx dy dw dh**************************************/
			/*		cout << endl;
			 cout << "dx, dy, dw, dh : " << endl;
			 cout << dx << ", " << dy << ", " << dw << ", " << dh << endl;*/
			/**************************************************************************************************/

			exp_score = exp(pred_cls[i * 5 + 1]);
			sum_exp_score = 0;
			for (int j = 0; j < 5; j++) {
				sum_exp_score += exp(pred_cls[i * 5 + j]);
			}

			pred_ctr_x = ctr_x + width * dx;
			pred_ctr_y = ctr_y + height * dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);

			pred_x1 = (pred_ctr_x - 0.5 * pred_w) / ratio_W;
			pred_y1 = (pred_ctr_y - 0.5 * pred_h) / ratio_H;
			pred_x2 = (pred_ctr_x + 0.5 * pred_w) / ratio_W;
			pred_y2 = (pred_ctr_y + 0.5 * pred_h) / ratio_H;
			/********************************************show x1, y1, x2, y2**************************************/
			/*		cout << endl;
			 cout << "x1, y1, x2, y2 : " << endl;
			 cout << pred_x1 << ", " << pred_y1 << ", " << pred_x2 << ", " << pred_y2 << endl;*/
			/**************************************************************************************************/

			pred[i * 5 + 0] = max(min(pred_x1, img_width -1), 0);
			pred[i * 5 + 1] = max(min(pred_y1, img_height -1), 0);
			pred[i * 5 + 2] = max(min(pred_x2, img_width -1), 0);
			pred[i * 5 + 3] = max(min(pred_y2, img_height -1), 0);
			pred[i * 5 + 4] = exp_score / sum_exp_score;
		} else {

			for (int j = 0; j < class_num; j++) {
				dx = box_deltas[i * 4 * (class_num + 1) + 4 * (j + 1) + 0];
				dy = box_deltas[i * 4 * (class_num + 1) + 4 * (j + 1) + 1];
				dw = box_deltas[i * 4 * (class_num + 1) + 4 * (j + 1) + 2];
				dh = box_deltas[i * 4 * (class_num + 1) + 4 * (j + 1) + 3];

				// bbox de-normalization
				dx *= 0.10f;
				dy *= 0.10f;
				dw *= 0.20f;
				dh *= 0.20f;

				exp_score = exp(pred_cls[i * (class_num + 1) + (j + 1)]);
				sum_exp_score = 0;
				for (int k = 0; k < (class_num + 1); k++) {
					sum_exp_score += exp(pred_cls[i * (class_num + 1) + k]);
				}

				pred_ctr_x = ctr_x + width * dx;
				pred_ctr_y = ctr_y + height * dy;
				pred_w = width * exp(dw);
				pred_h = height * exp(dh);

				pred_x1 = (pred_ctr_x - 0.5 * pred_w) / ratio_W;
				pred_y1 = (pred_ctr_y - 0.5 * pred_h) / ratio_H;
				pred_x2 = (pred_ctr_x + 0.5 * pred_w) / ratio_W;
				pred_y2 = (pred_ctr_y + 0.5 * pred_h) / ratio_H;

				pred[(j * num + i) * 5 + 0] = max(min(pred_x1, img_width -1),
						0);
				pred[(j * num + i) * 5 + 1] = max(min(pred_y1, img_height -1),
						0);
				pred[(j * num + i) * 5 + 2] = max(min(pred_x2, img_width -1),
						0);
				pred[(j * num + i) * 5 + 3] = max(min(pred_y2, img_height -1),
						0);
				pred[(j * num + i) * 5 + 4] = exp_score / sum_exp_score;

			}
		}
	}
}
bool compare_score(const Info& Info1, const Info& Info2)
{
    return Info1.score > Info2.score;
}
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
void Detector_mscnn::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
    vector<Info> my;
    Info tmp;
    for (int i = 0; i< num; i++)
    {
        tmp.score = pred[i*5 + 4];
        tmp.head = pred + i*5;
        my.push_back(tmp);
    }
    std::sort(my.begin(), my.end(), compare_score);
    for (int i=0; i<num; i++)
    {
        for (int j=0; j<5; j++)
            sorted_pred[i*5+j] = my[i].head[j];
    }
}


/** ===  FUNCTION  ======================================================================
*         Name:  vis_detections
*  Description:  Visuallize the detection result
* =====================================================================================*/

void Detector_mscnn::vis_detections(cv::Mat & image, int* keep, int & num_out, float* sorted_pred_cls, float & CONF_THRESH)
{
    int i=0;
    //cout << "num_out = " << num_out << endl;
    while(sorted_pred_cls[keep[i]*5+4]>CONF_THRESH && i < num_out)
    {
    	//cout << sorted_pred_cls[keep[i]*5+4] << endl;
/*        if(i>=num_out)
            return;*/
        if (car_or_not){
        	cv::rectangle(image,cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]),cv::Point(sorted_pred_cls[keep[i]*5+2], sorted_pred_cls[keep[i]*5+3]),cv::Scalar(0, 0, 255));
        	//cout << "car_i = " << i << endl;
        }else{
        	cv::rectangle(image,cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]),cv::Point(sorted_pred_cls[keep[i]*5+2], sorted_pred_cls[keep[i]*5+3]),cv::Scalar(255, 0, 0));
        	//cout << "ped_i = " << i << endl;
        }
        i++;

    }
}

/*void Detector_mscnn::vis_detections(cv::Mat image, int num, float* sorted_pred_cls, float CONF_THRESH)
{
   int i=0;
	for (int bbox = 0; bbox < num; bbox++) {
		if (sorted_pred_cls[bbox * 5 + 4] > CONF_THRESH) {
			if (car_or_not){
			cv::rectangle(image,
					cv::Point(sorted_pred_cls[bbox * 5 + 0],
							sorted_pred_cls[bbox * 5 + 1]),
					cv::Point(sorted_pred_cls[bbox * 5 + 2],
							sorted_pred_cls[bbox * 5 + 3]),
					cv::Scalar(0, 0, 255));
			}else{
				cv::rectangle(image,
						cv::Point(sorted_pred_cls[bbox * 5 + 0],
								sorted_pred_cls[bbox * 5 + 1]),
						cv::Point(sorted_pred_cls[bbox * 5 + 2],
								sorted_pred_cls[bbox * 5 + 3]),
						cv::Scalar(255, 0, 0));
			}
			i++;
		}
	}
}*/





vector<string> getFiles(string cate_dir, vector<string> & files_name) {
	vector<string> files;
	DIR *dir;
	struct dirent *ptr;
	if ((dir = opendir(cate_dir.c_str())) == NULL) {
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL) {
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) ///current dir OR parrent dir
			continue;
		else if (strstr(ptr->d_name, ".jpg") != NULL)    ///file
		{
			files.push_back(cate_dir + "/" + ptr->d_name);
			files_name.push_back(ptr->d_name);
		}
	}
	closedir(dir);
	sort(files.begin(), files.end());
	sort(files_name.begin(), files_name.end());
	return files;
}



int main(){

    // determine GPU
    int GPUID=0;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);

    string model_file = "/home/tianyuw/DNN/mscnn/examples/kitti_car/mscnn-7s-576-2x-trainval/Models/model_v2/mscnn_deploy.prototxt";
    string weights_file = "/home/tianyuw/DNN/mscnn/examples/kitti_car/mscnn-7s-576-2x-trainval/Models/model_v2/mscnn_ucar_trainval_2nd_iter_35000.caffemodel";
    // construct network detect vehicle
    Detector_mscnn det_mscnn_vehicle = Detector_mscnn(model_file, weights_file, 981, 540, true, 1, Detector_mscnn::VEHICLE);
     model_file = "/home/tianyuw/DNN/mscnn/examples/kitti_ped_cyc/mscnn-7s-576-2x-trainval/Models/model_v3/mscnn_deploy.prototxt";
     weights_file = "/home/tianyuw/DNN/mscnn/examples/kitti_ped_cyc/mscnn-7s-576-2x-trainval/Models/model_v3/mscnn_kitti_trainval_2nd_iter_35000.caffemodel";
    // construct network detect pedestrain and cyclist
    Detector_mscnn det_mscnn_ped_cyc = Detector_mscnn(model_file, weights_file, 1280, 720, false, 2, Detector_mscnn::PED_CYC);
    model_file = "/home/tianyuw/DNN/mscnn/examples/kitti_ped_cyc/mscnn-7s-576-2x-trainval/Models/model_v8/mscnn_deploy.prototxt";
    weights_file = "/home/tianyuw/DNN/mscnn/examples/kitti_ped_cyc/mscnn-7s-576-2x-trainval/Models/model_v8/mscnn_kitti_trainval_2nd_iter_35000.caffemodel";
    Detector_mscnn det_mscnn_TL = Detector_mscnn(model_file, weights_file, 1280, 720, false, 1, Detector_mscnn::TRAFFIC_LIGHT);
    // measure detection time
    double start_t, detect_time;
    // open a json file to save the result
    string json_path = "json_output.json";
    Write_Json json(json_path);
    // search image file path
    string image_dir = "/home/tianyuw/DNN/DATA/competition_data/demo";
    // output display image save path
    string save_path = "/home/tianyuw/cuda-workspace/ucar_evaluation_gpu/det_result";
    vector<string> image_file_path, image_file_name;
    image_file_path = getFiles(image_dir, image_file_name);
    int image_num = image_file_path.size();

    // image file
    Mat image;
    Show_Detection result;
    string image_name;
    for (vector<string>::iterator it = image_file_path.begin(); it != image_file_path.end(); ++it ){
    	start_t = cpuSecond();
    	image_name = (*it).substr((*it).find_last_of("/") + 1);
    	image = cv::imread(*it);
    	if (image.empty()) {
    		std::cout << "Can not get the image file !" << endl;
    		return 0;
    	}
    	result.disp_image = image.clone();
    	det_mscnn_vehicle.Detection(image, result, -100);

    	det_mscnn_ped_cyc.Detection(image, result, -100);

    	det_mscnn_TL.Detection(image, result, -100);

    	detect_time = cpuSecond() - start_t;
    	json.writeToJson(image_name, image_file_path.end() - it);
    	result.show_bboxes();
    	imwrite((save_path + "/" + image_name), result.disp_image);

    	// for demostration rescale disp_image
    	resize(result.disp_image, result.disp_image, Size(960, 540), 0, 0, INTER_LINEAR );
    	namedWindow("detection_result");
    	moveWindow("detection_result", 400, 300);
    	imshow("detection_result", result.disp_image);
    	waitKey(1000);
    	image.release();
    	result.disp_image.release();

    	cout << image_name << " : ";
    	cout << "detection time -> " << detect_time << " s" << endl;

    	result.clear_bboxes();
    }
    //close the json file
    json.~Write_Json();
return 0;
}
