### **SSC-SB: Semantic Similarity Contrast-based Street Block Monitoring**  

This repository contains the implementation of **SSC-SB (Semantic Similarity Contrast-based Street Block Monitoring)**, a contrastive learning-based approach for monitoring street block development and renewal using **Sentinel-2 time series imagery**.  

![image](https://github.com/user-attachments/assets/985c36f0-1a70-4351-b706-9b04ac1a5b52)


---

## **Pipeline**  

### **1. Train the Model**  
To train the model from scratch, run the following command:  
```bash
python modelTrain.py
```  

### **2. Run SSC-SB for Street Block Monitoring**  
Once the model is trained or pre-trained weights are downloaded, run:  
```bash
python blockMonitoring.py
```  
This will generate the SSC-SB monitoring results.  
![image](https://github.com/user-attachments/assets/26c1655e-166d-409d-8046-1910f0dfd7de)

![image](https://github.com/user-attachments/assets/efa0e060-b8d7-43cc-aef9-8833b653770b)

---

## **Dataset and Pre-trained Model**  

- **Changsha Sentinel-2 Data (2019-2024, Preprocessed & Normalized)**  
  📥 **Download**: [Google Drive](https://drive.google.com/drive/folders/1hnrXD46nRU7I2mOuMeslfeVpVbHYkG_t?usp=drive_link)  
  📂 **Place in**: `dataset/`  

- **Pre-trained SSC-SB Model (Trained on Changsha Data)**  
  📥 **Download**: [Google Drive](https://drive.google.com/drive/folders/1D_Em8llo4khvGnAi6kWch6oTjN4t9iRP?usp=drive_link)  
  📂 **Place in**: `save_model/`  

---

## **Publication & Future Plans**  

The paper is currently under review and will be published soon. Upon publication, we will release the full dataset.  

Stay tuned and thank you for your interest in SSC-SB!  

---

## **Acknowledgments**  

We would like to thank the following resources for their contributions to contrastive learning methodologies:  

- [The AI Summer – SimCLR](https://theaisummer.com/simclr/)  
- [TSCP2 by Cruise Research Group](https://github.com/cruiseresearchgroup/TSCP2)  

For any inquiries or collaboration opportunities, feel free to reach out! 🚀
