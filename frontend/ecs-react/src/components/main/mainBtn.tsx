import { useRecoilState } from "recoil"
import button from "../../assets/image/kakao_button.png"
import { isLog } from "../../recoil/atoms/userState"
export default function MainBtn() {
  const isLoged = useRecoilState(isLog)

  return (
    <div
      style={{ textAlign: "center", marginTop: "30px" }}
      onClick={() => {
        window.location.href =
          "http://192.168.100.207:8080/api/oauth2/authorization/kakao"
      }}
    >
      <img src={button} alt='' style={{ cursor: "pointer" }}></img>
    </div>
  )
}