package ai.certifai.solution.segmentation.cell;

import ai.certifai.utilities.Visualization;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import javax.swing.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;

public class MySegmentationModel {
//     TODO
//     1. Import UNET from model Zoo
//     2. Print Model Summary
//     3. Setup Training UI
//     4. Construct the datasetiterator

    private static final int seed = 42;
    private static final Random rng = new Random(seed);
    private static final int nEpochs = 1;
    private static final int height = 1;
    private static final int width = 1;
    private static final int batchSize = 4;
    private static final double trainPerc = 0.7;

    public static void main(String[] args) throws IOException {


        // 1. Import Zoo Model
        ZooModel zooModel = UNet.builder().build();

        // 2. Print Model Summary
        ComputationGraph unet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
        System.out.println(unet.summary());

        // 3. Set Listeners
        StatsStorage statsStorage = new InMemoryStatsStorage();
        StatsListener statsListener = new StatsListener(statsStorage);
        ScoreIterationListener scoreIterationListener = new ScoreIterationListener(1);

        // Initialize training UI
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);

        // 4. Construct the datasetiterator
        // Data Preparation
        CellDataSetIterator.setup(batchSize, 0.8, getImageTransform());
        RecordReaderDataSetIterator trainIter = CellDataSetIterator.trainIterator();
        RecordReaderDataSetIterator valIter = CellDataSetIterator.valIterator();

        // 1. Fine tuning config
        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .updater(new Adam(new StepSchedule(ScheduleType.EPOCH, 3e-4, 0.5, 5)))
                .seed(seed)
                .build();

        // 2. Modify the existing model
        ComputationGraph unetTransfer = new TransferLearning.GraphBuilder(unet)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("conv2d_4")
                .nInReplace("conv2d_1", 1, WeightInit.XAVIER)
                .nOutReplace("conv2d_23", 1, WeightInit.XAVIER)
                .removeVertexAndConnections("activation_23")
                .addLayer("output",
                        new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                .activation(Activation.SIGMOID).build(),
                        "conv2d_23")
                .setOutputs("output")
                .build();

        System.out.println(unetTransfer.summary());

        unetTransfer.setListeners(statsListener, scoreIterationListener);

        // VISUALISATION -  validation
        JFrame frameVal = Visualization.initFrame("Viz");
        JPanel panelVal = Visualization.initPanel(
                frameVal,
                batchSize,
                height,
                width,
                1
        );

        for (int i=0; i<1; i++) {
            while (trainIter.hasNext()) {
                DataSet trainDataSet = trainIter.next();
                unetTransfer.fit(trainDataSet);

                INDArray predict = unetTransfer.output(trainDataSet.getFeatures()) [0];

                Visualization.visualize(
                        trainDataSet.getFeatures(),
                        trainDataSet.getLabels(),
                        predict,
                        frameVal,
                        panelVal,
                        batchSize,
                        224,
                        224
                );
            }

            trainIter.reset();
        }

        // Evaluation
        Evaluation eval = new Evaluation(2);


        float IOUtotal = 0;
        int count = 0;
        while (valIter.hasNext()) {
            count++;
            DataSet valDataSet = valIter.next();

            INDArray predict = unetTransfer.output(valDataSet.getFeatures())[0];
            INDArray labels = valDataSet.getLabels();

            eval.eval(labels, predict);

            // IoU = TP/ (TP + FN + FP)
            float IOUCell = eval.truePositives().get(1) / (eval.truePositives().get(1) + eval.falseNegatives().get(1) + eval.falsePositives().get(1));
            float IOUBackground = eval.truePositives().get(0) / (eval.truePositives().get(0) + eval.falseNegatives().get(0) + eval.falsePositives().get(0));
            float IOU_ = (IOUCell + IOUBackground)/2;
            IOUtotal = IOUtotal + IOUCell;

            System.out.println(IOUtotal);
        }

        System.out.println("Mean IOU: " + IOUtotal / count);

    }

    public static ImageTransform getImageTransform() {
        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }

}
