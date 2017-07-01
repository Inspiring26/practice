// 这个程序配合xpdfbin文件，在windows上可提取PDF内容，我并没亲自测试过，也没在Mac上测试过



import java.io.*; 

public class Get_pdf { 
 public static void main(String args[]) throws Exception 
 { 
   String PATH_TO_XPDF="F:\\xpdfbin-win-3.04\\bin64\\pdftotext.exe"; 
   String filename="E:\\Android\\workpace\\java\\POI\\src\\com\\get\\java\\115152s8memmep5s3rmh1v.pdf"; 
   String[] cmd = new String[] { PATH_TO_XPDF, "-enc", "UTF-8", "-q", filename, "-"}; 
   Process p = Runtime.getRuntime().exec(cmd); 
   BufferedInputStream bis = new BufferedInputStream(p.getInputStream()); 
   InputStreamReader reader = new InputStreamReader(bis, "UTF-8"); 
   StringWriter out = new StringWriter(); 
   char [] buf = new char[10000]; 
   int len; 
   while((len = reader.read(buf))>= 0) { 
   //out.write(buf, 0, len); 
   System.out.println("the length is "+len); 
   } 
   reader.close(); 
   String ts=new String(buf); 
   System.out.println(ts); 
 } 
}

