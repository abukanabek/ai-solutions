# 📘 Kazakhstan Olympiad AI — Home Task

## 🔗 Resources

- **Problem Repository:**  
  https://github.com/yamazakiji/KazakhstanOlympiadAI-HomeTask

- **Dataset Repository:**  
  https://huggingface.co/datasets/myxik/KazOAI-2026-HomeTask

---

## 💡 Solution Overview

The key observation is that the provided dataset is missing **2 out of the 5 required formats**.  
To ensure the model performs well on the test set, these missing formats must be **generated manually**.

---

## ⚙️ Training Notes

- Make sure the `max_new_tokens` parameter is **not set too low**.  
  Otherwise, the model may **fail on certain samples**.

---

## 📊 My Approach

- Generated approximately **70,000 training samples**
- Trained for **1 epoch (~9 hours)**

> ⚠️ **Note:** This setup is **overkill**. A solid-performing model can be trained with **significantly less time and data**.

---

## ✅ Conclusion

That’s essentially it — straightforward once you handle the missing formats and training constraints properly.
