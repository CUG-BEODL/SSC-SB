### **SSC-SB: Semantic Similarity Contrast-based Street Block Monitoring**  

This repository contains the implementation of **SSC-SB (Semantic Similarity Contrast-based Street Block Monitoring)**, a contrastive learning-based approach for monitoring street block development and renewal using **Sentinel-2 time series imagery**.  

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
