import { useNavigate } from "react-router-dom"
import { parseJwt } from "../services/jwtDecode"
import Content from "../components/main/content2"
import MainBtn from "../components/main/mainBtn"
import Title from "../components/main/title"
import style from "../styles/main/main.module.css"
import { useEffect } from "react"
import { Cookies } from "react-cookie"
import { isLog, userName, userNo } from "../recoil/atoms/userState"
import { useRecoilState, useSetRecoilState } from "recoil"

export default function Main() {
  const setUserNo = useSetRecoilState(userNo)
  const setUserName = useSetRecoilState(userName)
  const [log, setLog] = useRecoilState(isLog)

  const navigate = useNavigate()
  const cookies = new Cookies()

  useEffect(() => {
    const token = cookies.get("accessToken")
    if (token !== undefined) {
      const obj = parseJwt(token)
      setUserNo(obj.no)
      setUserName(obj.name)
      setLog(true)
    }
  }, [])
  return (
    <div className={style.continer}>
      <div className={style.wrapper}>
        <Title></Title>
        <Content></Content>
        {log ? (
          <div>
            <div className={style.btnBox}>
              <button
                className={style.btn}
                onClick={() => {
                  navigate("/setting")
                }}
              >
                시작하기
              </button>
            </div>
          </div>
        ) : (
          <div className={style.btnContainer}>
            <MainBtn></MainBtn>
          </div>
        )}
      </div>
    </div>
  )
}
