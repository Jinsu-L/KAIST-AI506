import java.io.IOException;
import java.util.StringTokenizer;
import java.util.ArrayList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class LSH {
    /*
    public static class ExampleMapper extends Mapper<Object, Text, KEY_TYPE1, VALUE_TYPE1>{
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // write your own code
            String[] tokens = value.toString().split("\t");
            Text curr_key = new Text(tokens[0]);
            context.write(KEY_TYPE1_OBJECT, VALUE_TYPE1_OBJECT);
            context.write(curr_key, new Text("value")); // example
        }
    }

    public static class ExampleReducer extends Reducer<KEY_TYPE1, VALUE_TYPE1, KEY_TYPE2, VALUE_TYPE2> {
        public void reduce(KEY_TYPE1 key, Iterable<VALUE_TYPE1> values, Context context) throws IOException, InterruptedException {
            // write your own code
            for (VALUE_TYPE1 value : values) {
                break;
            }
            context.write(KEY_TYPE2_OBJECT, VALUE_TYPE2_OBJECT);
            context.write(key, new Text("value")); // example
        }
    }
    */
    public static class Subtask1Mapper extends Mapper<Object, Text, Text, Text>{
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // write your own code
            String[] tokens = value.toString().split("\t");
            Text curr_key = new Text(tokens[0]);
            context.write(KEY_TYPE1_OBJECT, VALUE_TYPE1_OBJECT);
            context.write(curr_key, new Text("value")); // example
        }
    }

    public static class Subtask1Reducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(KEY_TYPE1 key, Iterable<VALUE_TYPE1> values, Context context) throws IOException, InterruptedException {
            // write your own code
            for (VALUE_TYPE1 value : values) {
                break;
            }
            context.write(KEY_TYPE2_OBJECT, VALUE_TYPE2_OBJECT);
            context.write(key, new Text("value")); // example
        }
    }

    public static class Subtask2Mapper extends Mapper<Object, Text, Text, Text>{
        final static long[][] mod = {{597486750, 413963616}, {1040166821, 823616117}, {671531367, 700375935}, {372292445, 69238393}, {449175553, 466725910}, {1132247199, 360843279}, {743123192, 1105219197}, {838421488, 1216226950}, {1041947951, 596026982}, {744557697, 355087488}};

        private static long wordHash(String s){
            int len = s.length();
            long ret = 0;
            for(int i=len-1;i>=0;i--){
                ret = ((ret * 31) + (int)(s.charAt(i) - 'a') + 1) % 1234567891;
            }
            return ret;
        }

        private static String hashFunction(int func_id, String first_word, String second_word, String third_word){
            // hashFunction(2, 'i', 'have', 'a') -> 2nd hash function value of 3-shingle ('i', 'have', 'a')
            // 1 <= func_id <= 10
            return Long.toString((mod[func_id - 1][0] * wordHash(first_word) + mod[func_id - 1][1] * wordHash(second_word) + wordHash(third_word)) % 1234567891);
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // write your own code
            String[] tokens = value.toString().split("\t");
            Text curr_key = new Text(tokens[0]);
            context.write(KEY_TYPE1_OBJECT, VALUE_TYPE1_OBJECT);
            context.write(curr_key, new Text("value")); // example
        }
    }

    public static class Subtask2Reducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(KEY_TYPE1 key, Iterable<VALUE_TYPE1> values, Context context) throws IOException, InterruptedException {
            // write your own code
            for (VALUE_TYPE1 value : values) {
                break;
            }
            context.write(KEY_TYPE2_OBJECT, VALUE_TYPE2_OBJECT);
            context.write(key, new Text("value")); // example
        }
    }

    public static class Subtask3Mapper extends Mapper<Object, Text, Text, Text>{
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // write your own code
            String[] tokens = value.toString().split("\t");
            Text curr_key = new Text(tokens[0]);
            context.write(KEY_TYPE1_OBJECT, VALUE_TYPE1_OBJECT);
            context.write(curr_key, new Text("value")); // example
        }
    }

    public static class Subtask3Reducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(KEY_TYPE1 key, Iterable<VALUE_TYPE1> values, Context context) throws IOException, InterruptedException {
            // write your own code
            for (VALUE_TYPE1 value : values) {
                break;
            }
            context.write(KEY_TYPE2_OBJECT, VALUE_TYPE2_OBJECT);
            context.write(key, new Text("value")); // example
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        /*
        Job ExampleJob = Job.getInstance(conf, "EXAMPLE");
        ExampleJob.setJarByClass(LSH.class);
        ExampleJob.setNumReduceTasks(5);

        FileInputFormat.addInputPath(ExampleJob, new Path(args[0]));
        ExampleJob.setInputFormatClass(TextInputFormat.class);
        ExampleJob.setMapOutputKeyClass(KEY_TYPE1.class);
        ExampleJob.setMapOutputValueClass(VALUE_TYPE1.class);
        ExampleJob.setMapperClass(ExampleMapper.class);

        FileOutputFormat.setOutputPath(ExampleJob, new Path(args[0] + "/ex"));
        ExampleJob.setOutputFormatClass(TextOutputFormat.class);
        ExampleJob.setOutputKeyClass(KEY_TYPE2.class);
        ExampleJob.setOutputValueClass(VALUE_TYPE2.class);
        ExampleJob.setReducerClass(ExampleReducer.class);

        if(!ExampleJob.waitForCompletion(true)) System.exit(1);
        */
        
        // Run Subtask 1
        Job Subtask1Job = Job.getInstance(conf, "Subtask1");
        Subtask1Job.setJarByClass(LSH.class);
        Subtask1Job.setNumReduceTasks(5);

        FileInputFormat.addInputPath(Subtask1Job, new Path(args[0]));
        Subtask1Job.setInputFormatClass(TextInputFormat.class);
        Subtask1Job.setMapOutputKeyClass(Text.class);
        Subtask1Job.setMapOutputValueClass(Text.class);
        Subtask1Job.setMapperClass(Subtask1Mapper.class);

        FileOutputFormat.setOutputPath(Subtask1Job, new Path(args[0] + "/subtask1"));
        Subtask1Job.setOutputFormatClass(TextOutputFormat.class);
        Subtask1Job.setOutputKeyClass(Text.class);
        Subtask1Job.setOutputValueClass(Text.class);
        Subtask1Job.setReducerClass(Subtask1Reducer.class);

        if(!Subtask1Job.waitForCompletion(true)) System.exit(1);
        
        // Run Subtask 2
        Job Subtask2Job = Job.getInstance(conf, "Subtask2");
        Subtask2Job.setJarByClass(LSH.class);
        Subtask2Job.setNumReduceTasks(5);

        FileInputFormat.addInputPath(Subtask2Job, new Path(args[0] + "/subtask1"));
        Subtask2Job.setInputFormatClass(TextInputFormat.class);
        Subtask2Job.setMapOutputKeyClass(Text.class);
        Subtask2Job.setMapOutputValueClass(Text.class);
        Subtask2Job.setMapperClass(Subtask2Mapper.class);

        FileOutputFormat.setOutputPath(Subtask2Job, new Path(args[0] + "/subtask2"));
        Subtask2Job.setOutputFormatClass(TextOutputFormat.class);
        Subtask2Job.setOutputKeyClass(Text.class);
        Subtask2Job.setOutputValueClass(Text.class);
        Subtask2Job.setReducerClass(Subtask2Reducer.class);

        if(!Subtask2Job.waitForCompletion(true)) System.exit(1);
        
        // Run Subtask 3
        Job Subtask3Job = Job.getInstance(conf, "Subtask3");
        Subtask3Job.setJarByClass(LSH.class);
        Subtask3Job.setNumReduceTasks(5);

        FileInputFormat.addInputPath(Subtask3Job, new Path(args[0] + "/subtask2"));
        Subtask3Job.setInputFormatClass(TextInputFormat.class);
        Subtask3Job.setMapOutputKeyClass(Text.class);
        Subtask3Job.setMapOutputValueClass(Text.class);
        Subtask3Job.setMapperClass(Subtask3Mapper.class);

        FileOutputFormat.setOutputPath(Subtask3Job, new Path(args[0] + "/output"));
        Subtask3Job.setOutputFormatClass(TextOutputFormat.class);
        Subtask3Job.setOutputKeyClass(Text.class);
        Subtask3Job.setOutputValueClass(Text.class);
        Subtask3Job.setReducerClass(Subtask3Reducer.class);

        if(!Subtask3Job.waitForCompletion(true)) System.exit(1);

        System.exit(0);
    }
}
