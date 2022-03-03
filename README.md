# Segmentation에 관련된 논문 구현  
## Segmentation이 무엇인가?  
<img src = "https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-7.42.16-PM.png">
  
- Object Detection이 이미지가 어디에 있는지 찾아내 bounding box를 그려주고 class까지 분류해준다면  
- Semantic Segmentation은 위 이미지처럼 이미지의 모든 픽셀의 class를 예측하는 작업이다  
- 이미지에 같은 class의 여러 object를 분류까지 하는 것은 Instance Segmentation이라고 한다  

## Segmentation의 활용 사례  
### 자율 주행  
<img src = "https://blogs.nvidia.co.kr/wp-content/uploads/sites/16/2017/02/ces-computer-vision-example-web-398x256.gif">  

- 위 이미지처럼 실시간으로 Segmentation을 수행해 각 class를 구분해 안전하게 자율 주행을 할 수 있도록 도와준다  

### 의료 이미지  
<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2021/09/BRaTS-fig-1.png">  

- CT나 MRI를 통해 얻은 이미지 데이터에서 종양 등을 찾아내기도 한다  
- 혹은 세포를 분할해내기도 한다  
- 위 이미지는 분할을 통해 뇌에 존재하는 종양을 Segmentation을 통해 찾아낸 것이다
