/*三个必须的jar包
commons-collections4-4.1.jar
poi-3.16.jar
poi-scratchpad-3.16.jar
*/

// 本程序的功能是获取word的内容，这个程序是针对doc格式的
import java.io.File;
import java.io.FileInputStream;
import org.apache.poi.hwpf.extractor.WordExtractor;


public class ReadDoc{
	public static void main(String[] args) {
		String text = ""; 
		try{
			File file = new File("2016.doc");
			FileInputStream stream = new FileInputStream(file);
			WordExtractor word = new WordExtractor(stream);
			text = word.getText();
			System.out.println(text);
			
		}catch(Exception e){
			e.printStackTrace(); 
		}

	}
}