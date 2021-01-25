package ai.certifai.solution.object_detection.ChiliGradingDetector;

import ai.certifai.Helper;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class ChiliDataSetIterator {

    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static Path trainDir, testDir;
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yoloWidth = 416;
    public static final int yoloHeight = 416;

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, Path dir, int batchSize) throws IOException {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloHeight, yoloWidth, nChannels, gridHeight, gridWidth, new VocLabelProvider(dir.toString()));
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws IOException {
        return makeIterator(trainData, trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws IOException {
        return makeIterator(testData, testDir, batchSize);
    }

    public static void setup() throws IOException {
        loadData();
        trainDir = Paths.get(dataDir, "chili", "train");
        testDir = Paths.get(dataDir, "chili", "test");
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
    }

    private static void loadData() throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();
    }
}
