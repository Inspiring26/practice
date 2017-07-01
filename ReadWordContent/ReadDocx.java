/*三个必须的jar包
commons-collections4-4.1.jar
poi-3.16.jar
poi-scratchpad-3.16.jar
*/

// 本程序的功能是获取word的内容，这个程序是针对doc格式的
import java.io.File;
import java.io.FileInputStream;
import org.apache.poi.POIXMLTextExtractor;
import org.apache.poi.POIXMLDocument;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.apache.poi.openxml4j.opc.OPCPackage;


public class ReadDocx{
	public static void main(String[] args) {
		String text = ""; 
		try{
			// File file = new File("2016.doc");
			OPCPackage opcPackage = POIXMLDocument.openPackage("2016.docx");
			POIXMLTextExtractor extractor = new XWPFWordExtractor(opcPackage);
			// FileInputStream stream = new FileInputStream(file);
			// WordExtractor word = new WordExtractor(stream);
			String text2007 = extractor.getText();
			System.out.println(text2007);
			// text = word.getText();
			// System.out.println(text);
			
		}catch(Exception e){
			e.printStackTrace(); 
		}

	}
}