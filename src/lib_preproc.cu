/*
    Data preparation cuda functions
    Adapted to use with Python and numpy
    Author: Aloísio Dourado (jun, 2018)

    Credits to Shuran Song: Some functions were adapted from Caffe Code: see https://github.com/shurans/sscnet
*/

//nvcc --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_preproc.cu

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

#define NORMALS_OFFSET (2)

using namespace std;
using namespace std::chrono;

typedef high_resolution_clock::time_point clock_tick;

// Camera information
int frame_width = 640; // in pixels
int frame_height = 480;
float vox_unit = 0.02;
float vox_margin = 0.24;
float floor_high = 4.0;
int NUM_THREADS=128;
int DEVICE = 0;
float *parameters_GPU;
float sample_neg_obj_ratio=1;
int debug = 0;
int normals_offset = 3;

#define NUM_CLASSES (256)
#define NUM_PRIOR_CLASSES (11)
#define MAX_DOWN_SIZE (1000)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert (Error %d): %s File: %s Line: %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//float cam_K[9] = {518.8579f, 0.0f, (float)frame_width / 2.0f, 0.0f, 518.8579f, (float)frame_height / 2.0f, 0.0f, 0.0f, 1.0f};

float *cam_K;

float cam_info[27];

float *create_parameters_GPU(){

  float parameters[14];
  for (int i = 0; i<9; i++)
     parameters[i]=cam_K[i];
  parameters[9]  = frame_width;
  parameters[10]  = frame_height;
  parameters[11] = vox_unit;
  parameters[12] = vox_margin;
  parameters[13] = floor_high;

  float *parameters_GPU;

  cudaMalloc(&parameters_GPU, 14 * sizeof(float));
  cudaMemcpy(parameters_GPU, parameters, 14 * sizeof(float), cudaMemcpyHostToDevice);

  return (parameters_GPU);

}

clock_tick start_timer(){
    return (high_resolution_clock::now());
}

void end_timer(clock_tick t1, const char msg[]) {
  if (debug==1){
      clock_tick t2 = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
      printf("%s: %ld(ms)\n", msg, duration);
  }
}

void setup_CPP(int device, int num_threads, float *K, int fw, int fh, float v_unit,
               float v_margin, float f_high, int debug_flag){
    cam_K = K;
    DEVICE = device;
    NUM_THREADS = num_threads;
    frame_width = fw; // in pixels
    frame_height = fh;
    vox_unit = v_unit;
    vox_margin = v_margin;
    floor_high = f_high;

    cudaDeviceProp deviceProperties;
    gpuErrchk(cudaGetDeviceProperties(&deviceProperties, DEVICE));
    cudaSetDevice(DEVICE);

    parameters_GPU = create_parameters_GPU();

    if (debug_flag==1) {

        printf("\nUsing GPU: %s - (device %d)\n", deviceProperties.name, DEVICE);
        printf("Threads per block: %d\n", NUM_THREADS);
    }

    debug = debug_flag;



}


__device__
void get_parameters_GPU(float *parameters_GPU,
                         float **cam_K_GPU, int *frame_width_GPU, int *frame_height_GPU,
                         float *vox_unit_GPU, float *vox_margin_GPU, float *floor_high_GPU   ){
    *cam_K_GPU = parameters_GPU;
    *frame_width_GPU = int(parameters_GPU[9]);
    *frame_height_GPU = int(parameters_GPU[10]);
    *vox_unit_GPU = parameters_GPU[11];
    *vox_margin_GPU = parameters_GPU[12];
    *floor_high_GPU = parameters_GPU[13];
}


void destroy_parameters_GPU(float *parameters_GPU){

  cudaFree(parameters_GPU);

}

__device__
int modeLargerZero(const int *values, int size) {
  int count_vector[NUM_CLASSES] = {0};

  for (int i = 0; i < size; ++i)
      if  (values[i] > 0)
          count_vector[values[i]]++;

  int md = 0;
  int freq = 0;

  for (int i = 0; i < NUM_CLASSES; i++)
      if (count_vector[i] > freq) {
          freq = count_vector[i];
          md = i;
      }
  return md;
}

// find mode of in an vector
__device__
int mode(const int *values, int size) {
  int count_vector[NUM_CLASSES] = {0};

  for (int i = 0; i < size; ++i)
          count_vector[values[i]]++;

  int md = 0;
  int freq = 0;

  for (int i = 0; i < NUM_CLASSES; i++)
      if (count_vector[i] > freq) {
          freq = count_vector[i];
          md = i;
      }
  return md;
}

__global__
void Downsample_Kernel( int *in_vox_size, int *out_vox_size,
                        unsigned char *in_labels, float *in_tsdf, float *in_prior, unsigned char *in_grid_GPU,
                        unsigned char *out_labels, float *out_tsdf, float *out_prior,
                        int *out_scale, unsigned char *out_grid_GPU) {
    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    //if (vox_idx <= 10){
    //printf("dowsample kernel \n" );
    //}


    int label_downscale = *out_scale;


    if (vox_idx >= out_vox_size[0] * out_vox_size[1] * out_vox_size[2]){
      return;
    }

    //if (vox_idx<10)
    //   printf("out_vox_size %d %d %d %d\n", out_vox_size[0], out_vox_size[1], out_vox_size[2],  out_vox_size[0]* out_vox_size[1]* out_vox_size[2]);

    int down_size = label_downscale * label_downscale * label_downscale;

    //printf("down_size %d\n",down_size);

    //int emptyT = int((0.95 * down_size)); //Empty Threshold
    //int fullT = int((0.05 * down_size)); //Valid for priors
    int emptyT = int((0.95 * down_size)); //Empty Threshold
    int fullT = int((0.2 * down_size)); //Valid for priors

    int z = (vox_idx / ( out_vox_size[0] * out_vox_size[1]))%out_vox_size[2] ;
    int y = (vox_idx / out_vox_size[0]) % out_vox_size[1];
    int x = vox_idx % out_vox_size[0];


    int label_vals[MAX_DOWN_SIZE] = {0};
    int count_vals=0;
    float tsdf_val = 0;

    int num_255 =0;

    int zero_count = 0;
    int zero_surface_count = 0;


    for (int tmp_x = x * label_downscale; tmp_x < (x + 1) * label_downscale; ++tmp_x) {
      for (int tmp_y = y * label_downscale; tmp_y < (y + 1) * label_downscale; ++tmp_y) {
        for (int tmp_z = z * label_downscale; tmp_z < (z + 1) * label_downscale; ++tmp_z) {
          int tmp_vox_idx = tmp_z * in_vox_size[0] * in_vox_size[1] + tmp_y * in_vox_size[0] + tmp_x;
          label_vals[count_vals] = int(in_labels[tmp_vox_idx]);
          count_vals += 1;

          if (in_labels[tmp_vox_idx] == 0 || in_labels[tmp_vox_idx] == 255) {
            if (in_labels[tmp_vox_idx]==255)
               num_255++;
            zero_count++;
          }
          if (in_grid_GPU[tmp_vox_idx] == 0 || in_labels[tmp_vox_idx] == 255) {
            zero_surface_count++;
          }

          tsdf_val += in_tsdf[tmp_vox_idx];
          out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)] += in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)]; //sum occurences
          for (int i=0; i< NUM_PRIOR_CLASSES; i++) {
              out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)+i+1] += in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)+i+1]; // probabilities ensemble
          }
		  
          if(in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)]==0){
            in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)]=1;
          }else{
            for (int i=0; i< NUM_PRIOR_CLASSES; i++) {
              in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)+i+1] /= in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)]; // probabilities ensemble
            }
            in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)]=0;
          }	
		  
          /*
          if (vox_idx  == 125997 ){
                  //printf("x:%d, y:%d, z:%d\n", x, y, z);
                  for (int i=0; i< NUM_PRIOR_CLASSES+1; i++) {
                      printf("%11.10f ",in_prior[tmp_vox_idx*(NUM_PRIOR_CLASSES+1)+i]);
                  }
                  printf("\n");
          }
          */

        }
      }
    }

    //if (x<55 && y<35 && z<55)
    if (zero_count > emptyT) {
      out_labels[vox_idx] = mode(label_vals, down_size);
      //out_labels[vox_idx] = float(mode(label_vals, down_size));
      //out_labels[vox_idx] = int(mode(label_vals, down_size));
    } else {
      out_labels[vox_idx] = modeLargerZero(label_vals, down_size); // object label mode without zeros
      //out_labels[vox_idx] = int(modeLargerZero(label_vals, down_size)); // object label mode without zeros
      //printf("vox_idx: %d out: %d out_labels[vox_idx] =", vox_idx, out_labels[vox_idx]);
    }


    if (zero_surface_count > emptyT) {
      out_grid_GPU[vox_idx] = 0;
    } else {
      out_grid_GPU[vox_idx] = 1;
    }

    out_tsdf[vox_idx] = tsdf_val /  down_size;

    if (out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)]>=fullT) {

        for (int i=0; i< NUM_PRIOR_CLASSES; i++) {
          out_prior[vox_idx*(NUM_PRIOR_CLASSES+1) + i + 1] /= out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)]; // probabilities ensemble
        }
        out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)] = 0.0; // not empty
    } else {
        for (int i=0; i< NUM_PRIOR_CLASSES; i++) {
          out_prior[vox_idx*(NUM_PRIOR_CLASSES+1) + i + 1] = 0.0; // empty
        }
        out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)] = 1.0; // empty
    }
    /*
    if (vox_idx  == 125997 ){//|| vox_idx  == 125997 || vox_idx  == 106727) {
      printf("%d: ", vox_idx);
      for (int i=0; i<NUM_PRIOR_CLASSES+1;i++)
         printf("%11.10f ", out_prior[vox_idx*(NUM_PRIOR_CLASSES+1)+i]);
      printf("\n");

    }
    */
}



int ReadVoxLabel_CPP(const std::string &filename,
                  float *vox_origin,
                  float *cam_pose,
                  int *vox_size,
                  int *segmentation_class_map,
                  unsigned char *segmentation_label_fullscale) {

                    //downsample lable
  clock_tick t1 = start_timer();

  // Open file
  std::ifstream fid(filename, std::ios::binary);

  end_timer(t1,"open");


  // Read voxel origin in world coordinates
  for (int i = 0; i < 3; ++i) {
    fid.read((char*)&vox_origin[i], sizeof(float));
  }
  end_timer(t1,"origin");

  // Read camera pose
  for (int i = 0; i < 16; ++i) {
    fid.read((char*)&cam_pose[i], sizeof(float));
    //printf("%6.2f\n",cam_pose[i]);
  }
  end_timer(t1,"pose");

  // Read voxel label data from file (RLE compression)
  std::vector<unsigned int> scene_vox_RLE;
  while (!fid.eof()) {
    int tmp;
    fid.read((char*)&tmp, sizeof(int));
    if (!fid.eof())
      scene_vox_RLE.push_back(tmp);
  }
  end_timer(t1,"read");

  // Reconstruct voxel label volume from RLE
  int vox_idx = 0;
  int object_count=0;
  for (size_t i = 0; i < scene_vox_RLE.size() / 2; ++i) {
    unsigned int vox_val = scene_vox_RLE[i * 2];
    unsigned int vox_iter = scene_vox_RLE[i * 2 + 1];
    //if (object_count<20 & vox_val>1 & vox_val < 255)
    //   printf("vox val %d vox_iter %d label %d\n", vox_val, vox_iter, segmentation_class_map[vox_val]);
    for (size_t j = 0; j < vox_iter; ++j) {
      if (vox_val == 255) {                        //255: Out of view frustum
        segmentation_label_fullscale[vox_idx] = 255; //12 classes 0 - 11 + 12=Outside room
      } else {
        segmentation_label_fullscale[vox_idx] = segmentation_class_map[vox_val];
        if(segmentation_label_fullscale[vox_idx]>3) {
          object_count++;
        };

      }
      vox_idx++;
    }
  }
  end_timer(t1,"voxel");
  if (debug==1) {
      printf("Object count %d\n",object_count);
      //printf("vox_idx diff: %d\n",vox_idx - (240*144*240));
  }
  return object_count;
}


void DownsampleLabel_CPP(int *vox_size,
                         int out_scale,
                         unsigned char *segmentation_label_fullscale,
                         float *vox_tsdf_fullscale,
                         float *vox_prior_full_GPU,
                         unsigned char *segmentation_label_downscale,
                         float *vox_prior_downscale,
                         float *vox_weights,unsigned char *vox_grid) {

  //downsample lable
  clock_tick t1 = start_timer();

  int num_voxels_in = vox_size[0] * vox_size[1] * vox_size[2];
  int num_voxels_down = num_voxels_in/(out_scale*out_scale*out_scale);


  int out_vox_size[3];

  float *vox_tsdf = new float[num_voxels_down];
  unsigned char *vox_grid_downscale = new unsigned char[num_voxels_down];

  out_vox_size[0] = vox_size[0]/out_scale;
  out_vox_size[1] = vox_size[1]/out_scale;
  out_vox_size[2] = vox_size[2]/out_scale;

    if (debug==1) {
      printf("Downsample - num_voxels_down: %d\n",num_voxels_down);
      printf("Downsample - out_vox_size: %d\n",out_vox_size[0]*out_vox_size[1]*out_vox_size[2]);
  }


  int *in_vox_size_GPU;
  int *out_vox_size_GPU;
  unsigned char *in_labels_GPU;
  unsigned char *out_labels_GPU;
  int *out_scale_GPU;
  float *in_tsdf_GPU;
  float *out_tsdf_GPU;
  float *out_prior_GPU;
  unsigned char *in_grid_GPU;
  unsigned char *out_grid_GPU;

  gpuErrchk(cudaMalloc(&in_vox_size_GPU, 3 * sizeof(int)));
  gpuErrchk(cudaMalloc(&out_vox_size_GPU, 3 * sizeof(int)));
  gpuErrchk(cudaMalloc(&in_labels_GPU, num_voxels_in * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&in_tsdf_GPU, num_voxels_in * sizeof(float)));
  gpuErrchk(cudaMalloc(&in_grid_GPU, num_voxels_in * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&out_labels_GPU, num_voxels_down * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&out_tsdf_GPU, num_voxels_down * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_prior_GPU, num_voxels_down * (NUM_PRIOR_CLASSES+1)* sizeof(float)));
  gpuErrchk(cudaMalloc(&out_grid_GPU, num_voxels_down * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&out_scale_GPU, sizeof(int)));


  gpuErrchk(cudaMemcpy(in_vox_size_GPU, vox_size,  3 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(out_vox_size_GPU, out_vox_size,  3 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(in_labels_GPU, segmentation_label_fullscale, num_voxels_in * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(out_labels_GPU, segmentation_label_downscale, num_voxels_down * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(in_tsdf_GPU, vox_tsdf_fullscale, num_voxels_in * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(out_prior_GPU, vox_prior_downscale, num_voxels_down * (NUM_PRIOR_CLASSES+1) * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(in_grid_GPU, vox_grid, num_voxels_in * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(out_scale_GPU, &out_scale, sizeof(int), cudaMemcpyHostToDevice));

  int BLOCK_NUM = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  Downsample_Kernel<<< BLOCK_NUM, NUM_THREADS >>>(in_vox_size_GPU, out_vox_size_GPU,
                                                  in_labels_GPU, in_tsdf_GPU, vox_prior_full_GPU, in_grid_GPU,
                                                  out_labels_GPU, out_tsdf_GPU, out_prior_GPU,
                                                  out_scale_GPU, out_grid_GPU);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  end_timer(t1,"Downsample duration");



  gpuErrchk(cudaMemcpy(segmentation_label_downscale, out_labels_GPU, num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vox_tsdf, out_tsdf_GPU, num_voxels_down * sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vox_prior_downscale, out_prior_GPU, num_voxels_down * (NUM_PRIOR_CLASSES+1) * sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vox_grid_downscale, out_grid_GPU, num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost));



  // Find number of occupied voxels
  // Save voxel indices of background
  // Set label weights of occupied voxels as 1
  int num_occ_voxels = 0; //Occupied voxels in occluded regions
  std::vector<int> bg_voxel_idx;


  memset(vox_weights, 0, num_voxels_down * sizeof(float));

  for (int i = 0; i < num_voxels_down; ++i) {
      if ((segmentation_label_downscale[i]) > 0 && (segmentation_label_downscale[i]<255)) {
          //Occupied voxels in the room
          num_occ_voxels++;
          if (vox_grid_downscale[i]==1) {
              //surface
              vox_weights[i] = -1.0;
           } else {
              //occupied non surface
              vox_weights[i] = 1.0;
           }
      } else if ((vox_tsdf[i] < 0) && (segmentation_label_downscale[i]<255)) {
                      vox_weights[i] = .5; // background voxels in unobserved region in the room
      } else if (segmentation_label_downscale[i] == 255){  //outside room
          segmentation_label_downscale[i] = 0;
      }

  }

  end_timer(t1,"Downsample duration + copy");

  cudaFree(in_vox_size_GPU);
  cudaFree(out_vox_size_GPU);
  cudaFree(in_labels_GPU);
  cudaFree(out_labels_GPU);
  cudaFree(in_tsdf_GPU);
  cudaFree(out_tsdf_GPU);
  cudaFree(out_prior_GPU);
  cudaFree(in_grid_GPU);
  cudaFree(out_grid_GPU);
  cudaFree(out_scale_GPU);

  delete [] vox_tsdf;
  delete [] vox_grid_downscale;

}

void getDepthData_cpp(unsigned char *depth_image, float *depth_data){
  unsigned short depth_raw;
  for (int i = 0; i < frame_height * frame_width; ++i) {
    depth_raw = ((((unsigned short)depth_image[i * 2 + 1]) << 8) + ((unsigned short)depth_image[i * 2 + 0]));
    depth_raw = (depth_raw << 13 | depth_raw >> 3);
    depth_data[i] = float((float)depth_raw / 1000.0f);
  }
}

__device__
void calcNormals(float rx1, float ry1, float rz1,
                 float rx2, float ry2, float rz2,
                 float rx3, float ry3, float rz3,
                 int *nx, int *ny, int *nz){
  float ux, uy, uz;
  float vx, vy, vz;
  float nfx, nfy, nfz;

  ux = rx2 - rx1;
  uy = ry2 - ry1;
  uz = rz2 - rz1;

  vx = rx3 - rx1;
  vy = ry3 - ry1;
  vz = rz3 - rz1;

  nfx = uy * vz - uz * vy;
  nfy = uz * vx - ux * vz;
  nfz = ux * vy - uy * vx;

  float normal_length = sqrtf(pow(nfx,2) + pow(nfy,2) + pow(nfz,2));

  *nx =  (int)abs(roundf((nfx/normal_length * 255)));
  *ny =  (int)abs(roundf((nfy/normal_length * 255)));
  *nz =  (int)abs(roundf((nfz/normal_length * 255)));

}

__global__
void getNormalsKernel(float *cam_pose, int *vox_size,  float *vox_origin, float *depth_data,
                unsigned char *labels_3d_gpu, unsigned char *normals, unsigned char *xyz, unsigned char *labels_2d_GPU,
                float *parameters_GPU){

  float *cam_K_GPU;
  int frame_width_GPU, frame_height_GPU;
  float vox_unit_GPU, vox_margin_GPU, floor_high_GPU;

  get_parameters_GPU(parameters_GPU, &cam_K_GPU, &frame_width_GPU, &frame_height_GPU,
                                     &vox_unit_GPU, &vox_margin_GPU,  &floor_high_GPU);


  // Get point in world coordinate
  // Try to parallel later

  // Get point in world coordinate
  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;

  // XYZ encoding -----------------------------------------------------------------------------------------------------
  if (pixel_x >=  frame_width_GPU || pixel_y >=  frame_height_GPU )
     return;

  float point_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];

  float point_cam[3] = {0};
  point_cam[0] =  (pixel_x - cam_K_GPU[2])*point_depth/cam_K_GPU[0];
  point_cam[1] =  (pixel_y - cam_K_GPU[5])*point_depth/cam_K_GPU[4];
  point_cam[2] =  point_depth;

  float point_base[3] = {0};

  point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
  point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
  point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

  point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

  int z = (int)floor((point_base[0] - vox_origin[0])/ vox_unit_GPU) + 0;
  int x = (int)floor((point_base[1] - vox_origin[1])/ vox_unit_GPU) + 0;
  int y = (int)floor((point_base[2] - vox_origin[2])/ vox_unit_GPU) +0;//+ int(floor_high_GPU);

  int label_vals[MAX_DOWN_SIZE] = {0};
  int count_vals = 0;
  int occup_count = 0;

  for (int xx=MAX(0,x-1);xx<MIN(x+2,vox_size[0]);xx++)
      for (int yy=MAX(0,y-1);yy<MIN(y+2,vox_size[1]);yy++)
        for (int zz=MAX(0,z-1);zz<MIN(z+2,vox_size[2]);zz++) {
            int vox_idx = zz * vox_size[0] * vox_size[1] + yy * vox_size[0] + xx;

            if (labels_3d_gpu[vox_idx]==255) {
                 label_vals[count_vals] = 0;
            } else {
                if (labels_3d_gpu[vox_idx]==0) {
                    label_vals[count_vals] = 0;
                } else {
                    label_vals[count_vals] = labels_3d_gpu[vox_idx];
                    occup_count ++;
                }
            }
            count_vals ++;
        }

  if (occup_count > 1) {
      labels_2d_GPU[pixel_y * frame_width_GPU + pixel_x] = modeLargerZero(label_vals, MAX_DOWN_SIZE);
  } else {
      labels_2d_GPU[pixel_y * frame_width_GPU + pixel_x] = mode(label_vals, MAX_DOWN_SIZE); // object label mode without zeros
  }


  // World coordinate to grid coordinate
  if (point_depth == 0) {
      xyz[3*(pixel_y * frame_width_GPU + pixel_x) + 0] = 0;
      xyz[3*(pixel_y * frame_width_GPU + pixel_x) + 1] = 0;
      xyz[3*(pixel_y * frame_width_GPU + pixel_x) + 2] = 0;
  } else {
      xyz[3*(pixel_y * frame_width_GPU + pixel_x) + 0] = (int)floor((point_base[0] - vox_origin[0])/ vox_unit_GPU);
      xyz[3*(pixel_y * frame_width_GPU + pixel_x) + 1] = (int)floor((point_base[1] - vox_origin[1])/ vox_unit_GPU);
      xyz[3*(pixel_y * frame_width_GPU + pixel_x) + 2] = (int)floor((point_base[2] - vox_origin[2])/ vox_unit_GPU) * 240/144;
  }

  //Normals ---------------------------------------------------------------------------------------------------------
  if (pixel_x < NORMALS_OFFSET || pixel_y < NORMALS_OFFSET ||
      pixel_x >=  (frame_width_GPU-NORMALS_OFFSET) || pixel_y >=  (frame_height_GPU-NORMALS_OFFSET) )
     return;
  int pixel_nx[12] = {pixel_x-NORMALS_OFFSET, pixel_x, pixel_x+NORMALS_OFFSET,
                     pixel_x-NORMALS_OFFSET, pixel_x+NORMALS_OFFSET, pixel_x,
                     pixel_x-NORMALS_OFFSET, pixel_x+NORMALS_OFFSET, pixel_x-NORMALS_OFFSET,
                     pixel_x-NORMALS_OFFSET, pixel_x+NORMALS_OFFSET, pixel_x+NORMALS_OFFSET};
  int pixel_ny[12] = {pixel_y-NORMALS_OFFSET, pixel_y+NORMALS_OFFSET, pixel_y-NORMALS_OFFSET,
                     pixel_y+NORMALS_OFFSET, pixel_y+NORMALS_OFFSET, pixel_y-NORMALS_OFFSET,
                     pixel_y+NORMALS_OFFSET, pixel_y, pixel_y-NORMALS_OFFSET,
                     pixel_y, pixel_y+NORMALS_OFFSET, pixel_y-NORMALS_OFFSET};
  float rx[12], ry[12], rz[12];
  int nx[4] = {0}, ny[4] = {0}, nz[4] = {0};

  for (int i = 0; i<12; i++){
      float point_depth = depth_data[pixel_ny[i] * frame_width_GPU + pixel_nx[i]];

      float point_cam[3] = {0};
      point_cam[0] =  (pixel_nx[i] - cam_K_GPU[2])*point_depth/cam_K_GPU[0];
      point_cam[1] =  (pixel_ny[i] - cam_K_GPU[5])*point_depth/cam_K_GPU[4];
      point_cam[2] =  point_depth;

      float point_base[3] = {0};

      point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
      point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
      point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

      point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
      point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
      point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

      rz[i] = (point_base[0] - vox_origin[0]);
      rx[i] = (point_base[1] - vox_origin[1]);
      ry[i] = (point_base[2] - vox_origin[2]);
  }

  calcNormals(rx[0], ry[0], rz[0], rx[1], ry[1], rz[1], rx[2], ry[2], rz[2], &nx[0], &ny[0], &nz[0]);
  calcNormals(rx[3], ry[3], rz[3], rx[4], ry[4], rz[4], rx[5], ry[5], rz[5], &nx[1], &ny[1], &nz[1]);
  calcNormals(rx[6], ry[6], rz[6], rx[7], ry[7], rz[7], rx[8], ry[8], rz[8], &nx[2], &ny[2], &nz[2]);
  calcNormals(rx[9], ry[9], rz[9], rx[10], ry[10], rz[10], rx[11], ry[11], rz[11], &nx[3], &ny[3], &nz[3]);

  normals[3*(pixel_y * frame_width_GPU + pixel_x) + 0] =  (nx[0] + nx[1] + nx[2] + nx[3])/4;
  normals[3*(pixel_y * frame_width_GPU + pixel_x) + 1] =  (ny[0] + ny[1] + ny[2] + ny[3])/4;
  normals[3*(pixel_y * frame_width_GPU + pixel_x) + 2] =  (nz[0] + nz[1] + nz[2] + nz[3])/4;

}




void ComputeNormals_CPP(float *cam_pose, int *vox_size,  float *vox_origin, unsigned char *depth_image,
                        unsigned char *labels_3d, unsigned char *normals, unsigned char *xyz, unsigned char *labels_2d) {
  //cout << "\nComputeTSDF_CPP\n";
  clock_tick t1 = start_timer();

  float *depth_data = new float[frame_height * frame_width];
  getDepthData_cpp(depth_image, depth_data);

  float *cam_pose_GPU,  *vox_origin_GPU, *depth_data_GPU;
  unsigned char *normals_GPU;
  unsigned char *xyz_GPU;
  int *vox_size_GPU;

  unsigned char *labels_3d_GPU;
  unsigned char *labels_2d_GPU;

  cudaMalloc(&cam_pose_GPU, 16 * sizeof(float));
  cudaMalloc(&vox_size_GPU, 3 * sizeof(int));
  cudaMalloc(&vox_origin_GPU, 3 * sizeof(float));

  cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(float));
  cudaMalloc(&labels_3d_GPU, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(unsigned char));
  cudaMalloc(&normals_GPU, 3* frame_height * frame_width * sizeof(unsigned char));
  cudaMalloc(&xyz_GPU, 3* frame_height * frame_width * sizeof(unsigned char));
  cudaMalloc(&labels_2d_GPU, frame_height * frame_width * sizeof(unsigned char));

  cudaMemcpy(cam_pose_GPU, cam_pose, 16 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(vox_origin_GPU, vox_origin, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(depth_data_GPU, depth_data, frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(labels_3d_GPU, labels_3d, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(normals_GPU, normals, 3* frame_height * frame_width * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(xyz_GPU, xyz, 3* frame_height * frame_width * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(labels_2d_GPU, labels_2d, frame_height * frame_width * sizeof(unsigned char), cudaMemcpyHostToDevice);

  end_timer(t1, "Prepare duration");

  t1 = start_timer();
  // from depth map to binaray voxel representation

  getNormalsKernel<<<frame_width,frame_height>>>(cam_pose_GPU, vox_size_GPU,  vox_origin_GPU, depth_data_GPU,
                                                 labels_3d_GPU, normals_GPU, xyz_GPU, labels_2d_GPU, parameters_GPU);
  cudaDeviceSynchronize();

  end_timer(t1,"depth2Grid duration");

  t1 = start_timer();

  cudaMemcpy(normals, normals_GPU, 3* frame_height * frame_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaMemcpy(xyz,xyz_GPU, 3* frame_height * frame_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaMemcpy(labels_2d,labels_2d_GPU, frame_height * frame_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);


  //delete [] vox_grid;
  delete [] depth_data;


  cudaFree(cam_pose_GPU);
  cudaFree(vox_size_GPU);
  cudaFree(vox_origin_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(normals_GPU);
  cudaFree(xyz_GPU);
  cudaFree(labels_2d_GPU);
  cudaFree(labels_3d_GPU);

  end_timer(t1,"closeup duration");

}

void GetNormals_CPP(const char *filename,
                 float *cam_pose,
                 int *vox_size,
                 float *vox_origin,
                 int *segmentation_class_map,
                 unsigned char *depth_data,
                 unsigned char *normals,
                 unsigned char *xyz,
                 unsigned char *labels_2d
                 ) {

    clock_tick t1 = start_timer();

    unsigned char *segmentation_label_fullscale;
    segmentation_label_fullscale= (unsigned char *) malloc((vox_size[0]*vox_size[1]*vox_size[2]) * sizeof(unsigned char *));

    //int object_count;
    //object_count =
    ReadVoxLabel_CPP(filename, vox_origin, cam_pose, vox_size, segmentation_class_map, segmentation_label_fullscale);
    end_timer(t1,"ReadVoxLabel_CPP");

    //if (object_count>0) {
        ComputeNormals_CPP(cam_pose, vox_size, vox_origin, depth_data, segmentation_label_fullscale, normals, xyz, labels_2d);
        end_timer(t1,"ComputeNormals_CPP");

    //}
    free(segmentation_label_fullscale);

}

__global__
void depth2Grid_prior(float *cam_pose, int *vox_size,  float *vox_origin, int *out_scale,
                      float *depth_data, float *prior_data,
                      unsigned char *vox_grid, float *vox_prior, float *parameters_GPU, int *depth_map_GPU){

  float *cam_K_GPU;
  int frame_width_GPU, frame_height_GPU;
  float vox_unit_GPU, vox_margin_GPU, floor_high_GPU;

  get_parameters_GPU(parameters_GPU, &cam_K_GPU, &frame_width_GPU, &frame_height_GPU,
                                     &vox_unit_GPU, &vox_margin_GPU, &floor_high_GPU);


  // Get point in world coordinate
  // Try to parallel later

  // Get point in world coordinate
  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;
  int pixel_index = pixel_y * frame_width_GPU + pixel_x;
    
  if (pixel_x >=  frame_width_GPU || pixel_y >=  frame_height_GPU )
     return;

  float point_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];
  int prior_data_idx = NUM_PRIOR_CLASSES * (pixel_y * frame_width_GPU + pixel_x);

  float point_cam[3] = {0};
  point_cam[0] =  (pixel_x - cam_K_GPU[2])*point_depth/cam_K_GPU[0];
  point_cam[1] =  (pixel_y - cam_K_GPU[5])*point_depth/cam_K_GPU[4];
  point_cam[2] =  point_depth;

  float point_base[3] = {0};

  point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
  point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
  point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

  point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

  //HIGH RESOLUTION VOX GRID FILLING (240x144x240)

  //printf("vox_origin: %f,%f,%f\n",vox_origin[0],vox_origin[1],vox_origin[2]);
  // World coordinate to HIGH RESOLUTION grid coordinate
  int z = (int)floor((point_base[0] - vox_origin[0])/ vox_unit_GPU) + 0;
  int x = (int)floor((point_base[1] - vox_origin[1])/ vox_unit_GPU) + 0;
  int y = (int)floor((point_base[2] - vox_origin[2])/ vox_unit_GPU) + int(floor_high_GPU);
  //printf("point_base: %f,%f,%f, %d,%d,%d, %d,%d,%d \n",point_base[0],point_base[1],point_base[2], z, x, y, vox_size[0],vox_size[1],vox_size[2]);

  // mark vox_out with 1.0
  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
      int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
      int vox_prior_idx = (NUM_PRIOR_CLASSES + 1) * (z * vox_size[0] * vox_size[1] + y * vox_size[0] + x);
      depth_map_GPU[pixel_index] = vox_idx;
      vox_grid[vox_idx] = 1;
      atomicAdd(&vox_prior[vox_prior_idx], 1.0f); //count the occurency
      for (int i=0; i< NUM_PRIOR_CLASSES; i++) {
          atomicAdd(&vox_prior[vox_prior_idx + i + 1], prior_data[prior_data_idx + i]);
      }
  }
}

__global__
void SquaredDistanceTransform(float *cam_pose, int *vox_size,  float *vox_origin, float *depth_data, unsigned char *vox_grid,
                              float *vox_tsdf, float *parameters_GPU) {

    float *cam_K_GPU = parameters_GPU;
    int frame_width_GPU= int(parameters_GPU[9]), frame_height_GPU= int(parameters_GPU[10]);
    float vox_unit_GPU= parameters_GPU[11], vox_margin_GPU = parameters_GPU[12];

    int search_region = (int)roundf(vox_margin_GPU/vox_unit_GPU);

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (vox_idx >= vox_size[0] * vox_size[1] * vox_size[2]){
      return;
    }

    if (vox_grid[vox_idx] ==1 ){
       vox_tsdf[vox_idx] = 0; //0
       return;
    }

    int z = (vox_idx / ( vox_size[0] * vox_size[1]))%vox_size[2] ;
    int y = (vox_idx / vox_size[0]) % vox_size[1];
    int x = vox_idx % vox_size[0];

    // Get point in world coordinates XYZ -> YZX
    float point_base[3] = {0};
    point_base[0] = float(z) * vox_unit_GPU + vox_origin[0];
    point_base[1] = float(x) * vox_unit_GPU + vox_origin[1];
    point_base[2] = float(y) * vox_unit_GPU + vox_origin[2];

    // Encode height from floor ??? check later

    // Get point in current camera coordinates
    float point_cam[3] = {0};
    point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
    point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
    point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
    point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] + cam_pose[2 * 4 + 0] * point_base[2];
    point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] + cam_pose[2 * 4 + 1] * point_base[2];
    point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] + cam_pose[2 * 4 + 2] * point_base[2];
    if (point_cam[2] <= 0) {
      vox_tsdf[vox_idx] = 1; //1
      return;
    }
    // Project point to 2D
    int pixel_x = roundf(cam_K_GPU[0] * (point_cam[0] / point_cam[2]) + cam_K_GPU[2]);
    int pixel_y = roundf(cam_K_GPU[4] * (point_cam[1] / point_cam[2]) + cam_K_GPU[5]);
    if (pixel_x < 0 || pixel_x >= frame_width_GPU || pixel_y < 0 || pixel_y >= frame_height_GPU){ // outside FOV
      //vox_tsdf[vox_idx] = GPUCompute2StorageT(-1.0);
      vox_tsdf[vox_idx] = 1;  //1
      return;
    }

    // Get depth
    float point_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];
    if (point_depth < float(0.5f) || point_depth > float(8.0f))
    {
      vox_tsdf[vox_idx] = 1; //1
      return;
    }
    if (roundf(point_depth) == 0){ // mising depth
      vox_tsdf[vox_idx] = -1;
      return;
    }


    // Get depth difference
    float point_dist = (point_depth - point_cam[2]) * sqrtf(1 + powf((point_cam[0] / point_cam[2]), 2) + powf((point_cam[1] / point_cam[2]), 2));
    //float sign = point_dist/abs(point_dist);

    float sign;
    if (abs(point_depth - point_cam[2]) < 0.0001){
        sign = 1; // avoid NaN
    }else{
        sign = (point_depth - point_cam[2])/abs(point_depth - point_cam[2]);
    }
    vox_tsdf[vox_idx] = sign;

    int radius=search_region; // out -> in
    int found = 0;
    //fixed y planes
    int iiy = max(0,y-radius);
    for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iiy = min(y+radius,vox_size[1]);
    for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    //fixed x planes
    int iix = max(0,x-radius);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iix = min(x+radius,vox_size[0]);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    //fixed z planes
    int iiz = max(0,z-radius);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iiz = min(z+radius,vox_size[2]);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }


    if (found == 0)
        return;

    radius=1; // in -> out
    found = 0;
    while (radius < search_region) {
        //fixed y planes
        int iiy = max(0,y-radius);
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iiy = min(y+radius,vox_size[1]);
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        //fixed x planes
        int iix = max(0,x-radius);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iix = min(x+radius,vox_size[0]);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        //fixed z planes
        int iiz = max(0,z-radius);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iiz = min(z+radius,vox_size[2]);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        if (found == 1)
          return;

        radius++;

    }

}


void ComputeTSDF_CPP(float *cam_pose, int *vox_size,  float *vox_origin, int out_scale,
                     unsigned char *depth_image, float *prior_data,
                     unsigned char *vox_grid, float *vox_tsdf, float *vox_prior_GPU, int *depth_map) {


  //cout << "\nComputeTSDF_CPP\n";
  clock_tick t1 = start_timer();

  float *depth_data = new float[frame_height * frame_width];
  getDepthData_cpp(depth_image, depth_data);

  int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];
  int num_pixels = 480 * 640;
  int vox_size_down[] = {vox_size[0]/out_scale, vox_size[1]/out_scale, vox_size[2]/out_scale};

  float *cam_pose_GPU,  *vox_origin_GPU, *depth_data_GPU, *vox_tsdf_GPU, *prior_data_GPU ;
  unsigned char *vox_grid_GPU;
  int *vox_size_GPU, *vox_size_down_GPU, *out_scale_GPU, *depth_map_GPU;



  gpuErrchk(cudaMalloc(&cam_pose_GPU, 16 * sizeof(float)));
  gpuErrchk(cudaMalloc(&vox_size_GPU, 3 * sizeof(int)));
  gpuErrchk(cudaMalloc(&depth_map_GPU, num_pixels * sizeof(int)));
  gpuErrchk(cudaMalloc(&vox_size_down_GPU, 3 * sizeof(int)));
  gpuErrchk(cudaMalloc(&vox_origin_GPU, 3 * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_scale_GPU, sizeof(int)));

  gpuErrchk(cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(float)));
  gpuErrchk(cudaMalloc(&prior_data_GPU, NUM_PRIOR_CLASSES * frame_height * frame_width * sizeof(float)));
  gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&vox_tsdf_GPU, num_voxels * sizeof(float)));

  gpuErrchk(cudaMemset(vox_tsdf_GPU, 0, num_voxels * sizeof(float)));
  gpuErrchk(cudaMemset(vox_grid_GPU, 0, num_voxels * sizeof(unsigned char)));
  gpuErrchk(cudaMemset(depth_map_GPU, 0, num_pixels * sizeof(int)));
  gpuErrchk(cudaMemcpy(out_scale_GPU, &out_scale, sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(cam_pose_GPU, cam_pose, 16 * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(depth_map_GPU, depth_map, num_pixels * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vox_size_down_GPU, vox_size_down, 3 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vox_origin_GPU, vox_origin, 3 * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(prior_data_GPU, prior_data, NUM_PRIOR_CLASSES * frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice));


  end_timer(t1, "Prepare duration");


  t1 = start_timer();
  // from depth map to binary voxel representation

  if (debug==1){
  printf("depth2Grid_prior<<<%d,%d>>>", frame_width,frame_height);
  }

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  depth2Grid_prior<<<frame_width,frame_height>>>(cam_pose_GPU, vox_size_GPU, vox_origin_GPU, out_scale_GPU,
                                           depth_data_GPU, prior_data_GPU,
                                           vox_grid_GPU, vox_prior_GPU, parameters_GPU, depth_map_GPU);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  end_timer(t1,"depth2Grid duration");

  int BLOCK_NUM = int((num_voxels + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  // distance transform

  t1 = start_timer();

  SquaredDistanceTransform<<< BLOCK_NUM, NUM_THREADS >>>(cam_pose_GPU, vox_size_GPU,  vox_origin_GPU, depth_data_GPU, vox_grid_GPU, vox_tsdf_GPU, parameters_GPU);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  end_timer(t1,"SquaredDistanceTransform");

  t1 = start_timer();

  gpuErrchk( cudaMemcpy(vox_grid, vox_grid_GPU, num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  end_timer(t1,"closeup duration a1");
  gpuErrchk( cudaMemcpy(vox_tsdf, vox_tsdf_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

  end_timer(t1,"closeup duration a");
  gpuErrchk( cudaMemcpy(depth_map, depth_map_GPU, num_pixels * sizeof(int), cudaMemcpyDeviceToHost));

  //delete [] vox_grid;
  delete [] depth_data;
  end_timer(t1,"closeup duration b");


  cudaFree(cam_pose_GPU);
  cudaFree(vox_size_GPU);
  cudaFree(vox_size_down_GPU);
  cudaFree(out_scale_GPU);
  cudaFree(vox_origin_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(prior_data_GPU);
  cudaFree(vox_grid_GPU);
  cudaFree(depth_map_GPU);
  //cudaFree(vox_prior_GPU);
  cudaFree(vox_tsdf_GPU);

  end_timer(t1,"closeup duration");

}


void FlipTSDF_CPP( int *vox_size, float *vox_tsdf){


  for (int vox_idx=0; vox_idx< vox_size[0]*vox_size[1]*vox_size[2]; vox_idx++) {

      float value = float(vox_tsdf[vox_idx]);
      if (value > 1)
          value =1;


      float sign;
      if (abs(value) < 0.001)
        sign = 1;
      else
        sign = value/abs(value);

      vox_tsdf[vox_idx] = sign*(max(0.001,(1.0-abs(value))));
  }
}


void Process_CPP(const char *filename,
                 float *cam_pose,
                 int *vox_size,
                 float *vox_origin,
                 int out_scale,
                 int *segmentation_class_map,
                 unsigned char *depth_data,
                 float *prior_data, //one-hot 2D segmentation probs
                 unsigned char *vox_grid,
                 float *vox_tsdf,
                 float *vox_prior, //3D projected segmentation probs
                 float *vox_weights,
                 unsigned char *segmentation_label_downscale,
                 int *depth_map,
                 float *vox_prior_full,
                 unsigned char *segmentation_label_full
                 ){
    clock_tick t1 = start_timer();

    int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    unsigned char *segmentation_label_fullscale;
    segmentation_label_fullscale= (unsigned char *) malloc((vox_size[0]*vox_size[1]*vox_size[2]) * sizeof(unsigned char));

    int object_count;

    object_count = ReadVoxLabel_CPP(filename, vox_origin, cam_pose, vox_size, segmentation_class_map, segmentation_label_fullscale);
    //printf("file %s\n", filename);
    //printf("object count %d\n", object_count);
    end_timer(t1,"ReadVoxLabel_CPP");

    float *in_labels_GPU;
    gpuErrchk(cudaMalloc(&in_labels_GPU, num_voxels * sizeof(unsigned char)));
    gpuErrchk(cudaMemset(in_labels_GPU, 0, num_voxels * sizeof(unsigned char)));

    if (object_count>0) {

        float *vox_prior_full_GPU;
        gpuErrchk(cudaMalloc(&vox_prior_full_GPU, (NUM_PRIOR_CLASSES + 1) * num_voxels * sizeof(float)));
        gpuErrchk(cudaMemset(vox_prior_full_GPU, 0, (NUM_PRIOR_CLASSES + 1) * num_voxels * sizeof(float)));

        t1 = start_timer();
        memset(vox_grid, 0, num_voxels * sizeof(unsigned char));
        ComputeTSDF_CPP(cam_pose, vox_size, vox_origin, out_scale, depth_data, prior_data, vox_grid, vox_tsdf, vox_prior_full_GPU, depth_map);

        gpuErrchk(cudaMemcpy(in_labels_GPU, segmentation_label_fullscale, num_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(segmentation_label_full, in_labels_GPU, num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        end_timer(t1,"ComputeTSDF_edges_CPP");

        t1 = start_timer();
        DownsampleLabel_CPP(vox_size,
                            out_scale,
                            segmentation_label_fullscale,
                            vox_tsdf,
                            vox_prior_full_GPU,
                            segmentation_label_downscale,
                            vox_prior,
                            vox_weights,vox_grid);

        gpuErrchk(cudaMemcpy(vox_prior_full, vox_prior_full_GPU, (NUM_PRIOR_CLASSES + 1) * num_voxels * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(vox_prior_full_GPU);
        end_timer(t1,"DownsampleLabel_CPP");

        t1 = start_timer();
        FlipTSDF_CPP( vox_size, vox_tsdf);
        end_timer(t1,"FlipTSDF_CPP");
    }
    t1 = start_timer();
    free(segmentation_label_fullscale);
    cudaFree(in_labels_GPU);
    end_timer(t1,"free");
    //FlipTSDF_CPP( out_vox_size, vox_vol);
}

extern "C" {
    void Process(const char *filename,
                  float *cam_pose,
                  int *vox_size,
                  float *vox_origin,
                  int out_scale,
                  int *segmentation_class_map,
                  unsigned char *depth_data,
                  float *prior_data,
                  unsigned char *vox_grid,
                  float *vox_tsdf,
                  float *vox_prior,
                  float *vox_weights,
                  unsigned char *segmentation_label_downscale,
                  int *depth_map,
                  float *vox_prior_full,
                  unsigned char *segmentation_label_full) {
                                 Process_CPP(filename,
                                             cam_pose,
                                             vox_size,
                                             vox_origin,
                                             out_scale,
                                             segmentation_class_map,
                                             depth_data,
                                             prior_data,
                                             vox_grid,
                                             vox_tsdf,
                                             vox_prior,
                                             vox_weights,
                                             segmentation_label_downscale,
                                             depth_map,
                                             vox_prior_full,
                                             segmentation_label_full) ;
                  }

    void GetNormals(const char *filename,
                  float *cam_pose,
                  int *vox_size,
                  float *vox_origin,
                  int *segmentation_class_map,
                  unsigned char *depth_data,
                  unsigned char *normals,
                  unsigned char *xyz,
                  unsigned char *labels_2d) {
                                 GetNormals_CPP(filename,
                                             cam_pose,
                                             vox_size,
                                             vox_origin,
                                             segmentation_class_map,
                                             depth_data,
                                             normals,
                                             xyz,
                                             labels_2d) ;
                  }

    void setup(int device, int num_threads, float *K, int fw, int fh, float v_unit, float v_margin,
               float floor_high, int debug_flag){
                                  setup_CPP(device, num_threads, K, fw, fh, v_unit, v_margin, floor_high, debug_flag);


    }
}