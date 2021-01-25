/*
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.certifai.training.image_processing;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 *
 * 1. Go to https://image.online-convert.com/, convert resources/image_processing/opencv.png into the following format:
 *       - .bmp
 *       - .jpg
 *       - .tiff
 *     Save them to the same resources/image_processing folder.
 *
 *  2. Use the .imread function to load each all the images in resources/image_processing,
 *       and display them using Display.display
 *
 *
 *  3. Print the following image attributes:
 *       - depth
 *       - number of channel
 *       - width
 *       - height
 *
 *  4. Repeat step 2 & 3, but this time load the images in grayscale
 *
 *  5. Resize file
 *
 *  6. Write resized file to disk
 *
 * */

public class LoadImages {
    public static void main(String[] args) throws IOException {

        String myImage = new ClassPathResource("image_processing/opencv.png").getFile().getAbsolutePath();
        Mat source = imread(myImage);
        Mat source2 = imread(myImage, IMREAD_GRAYSCALE);

        Display.display(source, "Original Image");
        Display.display(source2, "Greyscale Image");

        System.out.println("No. of Channels:" + source.channels());
        System.out.println("Image Width:" + source.arrayWidth());
        System.out.println("Image Height:" + source.arrayHeight());
        System.out.println("Image Depth:" + source.depth());
        System.out.println("Image Array Depth:" + source.arrayDepth());

        Mat downsize = new Mat();
        resize(source, downsize, new Size(500, 500));
        Display.display(downsize, "DownSampled");

        Mat upsize_nearest = new Mat();
        Mat upsize_linear = new Mat();
        Mat upsize_cubic = new Mat();
        resize(downsize, upsize_nearest, new Size(1478, 1200), 0, 0, INTER_NEAREST);
        resize(downsize, upsize_linear, new Size(1478, 1200), 0, 0, INTER_LINEAR);
        resize(downsize, upsize_cubic, new Size(1478, 1200), 0, 0, INTER_CUBIC);
        Display.display(upsize_nearest, "UpSampled Nearest Neighbour Interpolation");
        Display.display(upsize_linear, "UpSampled Bi-linear Interpolation");
        Display.display(upsize_cubic, "UpSampled Bi-cubic Interpolation");

        /*
        *
        * ENTER YOUR CODE HERE
        *
        * */


    }
}
