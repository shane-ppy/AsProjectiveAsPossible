// #include "SiftROCm.h"

// using namespace cv;
// using namespace std;

// void detectSiftMatchWithROCm(Mat &img1, Mat &img2, Eigen::MatrixXf &match)
// {
//     int *m = 0;
//     vl_uint8 *desc1 = 0, *desc2 = 0;
//     double *kp1 = 0, *kp2 = 0;
//     int nkp1 = detectSiftROCm(img1, kp1, desc1);
//     int nkp2 = detectSiftROCm(img2, kp2, desc2);
//     cout << "num kp1: " << nkp1 << endl;
//     cout << "num kp2: " << nkp2 << endl;

//     int nmatch = matchDescriptorWithRatioTest(desc1, desc2, nkp1, nkp2, m);
//     cout << "num match: " << nmatch << endl;
//     match.resize(nmatch, 6);
//     for (int i = 0; i < nmatch; i++) {
//         int index1 = m[i*2+0];
//         int index2 = m[i*2+1];
//         match.row(i) << kp1[index1*4+1], kp1[index1*4+0], 1, kp2[index2*4+1], kp2[index2*4+0], 1;
//     }

//     free(kp1);
//     free(kp2);
//     free(desc1);
//     free(desc2);
//     free(m);
// }

// typedef struct _sift_descriptor {
//     vl_uint8 descr[128];
// } sift_descriptor;

// int detectSiftROCm(Mat &img1, double* &kps, vl_uint8* &descr) {
//     vx_context context = vxCreateContext();
//     ERROR_CHECK_OBJECT(context);
//     vxRegisterLogCallback(context, log_callback, vx_false_e);
    
//     vxLoadKernels(context, "vx_opencv");
    
//     vx_graph graph = vxCreateGraph(context);
//     ERROR_CHECK_OBJECT(graph);

//     int width = 860, height = 860;
//     vx_image inter_luma = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
//     ERROR_CHECK_OBJECT(inter_luma);

//     Mat input, output;
//     cvtColor(img1, input, COLOR_RGB2GRAY);
//     // cvtColor(img1, output, COLOR_RGB2GRAY);
//     if (input.empty()) {
//         printf("Image not found\n");
//     }

//     // cv::resize(input, input, Size(width, height));
//     // cv::resize(output, output, Size(width, height));

//     vx_rectangle_t cv_image_region;
//     cv_image_region.start_x    = 0;
//     cv_image_region.start_y    = 0;
//     cv_image_region.end_x      = width;
//     cv_image_region.end_y      = height;
//     vx_imagepatch_addressing_t cv_image_layout;
//     cv_image_layout.stride_x   = 1;
//     cv_image_layout.stride_y   = input.step;
//     vx_uint8 * cv_image_buffer = input.data;
//     vx_enum siftDescType = vxRegisterUserStruct(context, 128);

//     ERROR_CHECK_STATUS( vxCopyImagePatch( inter_luma, &cv_image_region, 0,
//                                           &cv_image_layout, cv_image_buffer,
//                                           VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );
    
//     vx_array keypoints = vxCreateArray( context, VX_TYPE_KEYPOINT, 10000);
//     ERROR_CHECK_OBJECT( keypoints );
//     vx_array descriptors = vxCreateArray(context, siftDescType, 10000);
//     ERROR_CHECK_OBJECT(descriptors);

//     vx_int32 nFeatures = 0;
//     vx_int32 nOctaveLayers = 3;
//     vx_float32 contrastThreshold = 0.03;
//     vx_int32 edgeThreshold = 10;
//     vx_float32 sigma = 0.5;
//     vx_node nodes[] =
//     {
//         vxExtCvNode_siftDetect(graph, inter_luma, inter_luma, keypoints, nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma),
//         vxExtCvNode_siftCompute(graph, inter_luma, inter_luma, keypoints, descriptors, nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
//     };
    
//     for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
//     {
        
//         ERROR_CHECK_OBJECT( nodes[i] );
//         ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
//     }

//     ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );
//     ERROR_CHECK_STATUS( vxProcessGraph( graph ) );

    
//     vx_size num_corners = 0;
//     ERROR_CHECK_STATUS(vxQueryArray (keypoints, VX_ARRAY_NUMITEMS, &num_corners, sizeof(num_corners)));
//     if (num_corners > 0) {
//         vx_size kp_stride;
//         vx_map_id kp_map;
//         vx_uint8 * kp_buf;
//         ERROR_CHECK_STATUS(vxMapArrayRange (keypoints, 0, num_corners, &kp_map, &kp_stride, (void **)&kp_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
//         kps = (double*) malloc(4*sizeof(double)*num_corners);
//         for (vx_size nkp = 0; nkp < num_corners; nkp++) {
//             vx_keypoint_t * kp = (vx_keypoint_t *) (kp_buf + nkp*kp_stride);
//             // std::cout << kp->y+1 << ", " << kp->x+1 << ", " << kp->strength << ", " << kp->orientation  * M_PI / 180 - M_PI << std::endl;
//             kps[4*nkp+0] = kp->y+1;
//             kps[4*nkp+1] = kp->x+1;
//             kps[4*nkp+2] = kp->strength;
//             kps[4*nkp+3] = kp->orientation * M_PI / 180 - M_PI;
//         }
//     }

//     vx_size num_desc = 0;
//     ERROR_CHECK_STATUS(vxQueryArray (descriptors, VX_ARRAY_NUMITEMS, &num_desc, sizeof(num_desc)));
//     if (num_desc > 0) {
//         vx_size desc_stride;
//         vx_map_id desc_map;
//         vx_uint8 * desc_buf;
//         ERROR_CHECK_STATUS(vxMapArrayRange (descriptors, 0, num_desc, &desc_map, &desc_stride, (void **)&desc_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
//         descr = (vl_uint8*) malloc(128*sizeof(vl_uint8)*num_corners);
        
//         memcpy(descr, desc_buf, 128 * sizeof(vl_uint8) * num_corners);
//     }

//     // imshow( "OrbDetect", output );
//     // waitKey(0);


//     ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
//     ERROR_CHECK_STATUS( vxReleaseArray (&keypoints));
//     ERROR_CHECK_STATUS( vxReleaseArray(&descriptors))
//     ERROR_CHECK_STATUS( vxReleaseImage( &inter_luma ) );
//     ERROR_CHECK_STATUS( vxReleaseContext( &context ) );
//     return num_corners;
// }