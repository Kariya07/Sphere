import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;

import java.io.File;


public class CpFile {
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        String substr = ":///";
        int num = args.length;
        if (args[num-2].contains(substr) && args[num-1].contains(substr)){
            File srcFile = new File(args[num-2]);
            File destFile = new File(args[num-1]);
            FileUtils.copyFile(srcFile, destFile);
            System.out.println("cp "+args[num-2]+" "+args[num-1]);
            return;
        }
        if (args[num-2].contains(substr) && !(args[num-1].contains(substr))){
            File srcFile = new File(args[num-2]);
            FileUtil.copy(srcFile, fs, new Path(args[num-1]), false, conf);
            System.out.println("-put "+args[num-2]+" "+args[num-1]);
            return;
        }
        if (!(args[num-2].contains(substr)) && args[num-1].contains(substr)){
            File destFile = new File(args[num-1]);
            FileUtil.copy(fs, new Path(args[num-2]), destFile, false, conf);
            System.out.println("-get "+args[num-2]+" "+args[num-1]);
        }else{
            FileUtil.copy(fs, new Path(args[num-2]), fs, new Path(args[num-1]), false, conf);
            System.out.println("-cp "+args[num-2]+" "+args[num-1]);
        }
    }
}
