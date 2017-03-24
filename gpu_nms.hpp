/*
 * gpu_nms.hpp
 *
 *  Created on: Dec 14, 2016
 *      Author: tianyuw
 */

#ifndef GPU_NMS_HPP_
#define GPU_NMS_HPP_
void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);

#endif /* GPU_NMS_HPP_ */
