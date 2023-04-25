import style from "../../styles/drawing/palette.module.css"
export default function Palette({
  changeColor,
  changeSize,
}: {
  changeColor: any
  changeSize: any
}) {
  return (
    <div className={style.container}>
      <div className={style.penSize}>
        <button
          className={`${style.colorBtn} ${style.red}`}
          onClick={(e: any) => {
            changeColor("#ff0000")
          }}
        ></button>
        <button
          className={`${style.colorBtn} ${style.orange}`}
          onClick={() => {
            changeColor("#ff9900")
          }}
        ></button>

        <button
          className={`${style.colorBtn} ${style.yellow}`}
          onClick={() => {
            changeColor("#fef400")
          }}
        ></button>
        <button
          className={`${style.colorBtn} ${style.green}`}
          onClick={() => {
            changeColor("#01eb18")
          }}
        ></button>

        <button
          className={`${style.colorBtn} ${style.blue}`}
          onClick={() => {
            changeColor("#037fda")
          }}
        ></button>
        <button
          className={`${style.colorBtn} ${style.purple}`}
          onClick={() => {
            changeColor("#bf00cf")
          }}
        ></button>

        <button
          className={`${style.colorBtn} ${style.black}`}
          onClick={() => {
            changeColor("black")
          }}
        ></button>
        <button
          className={`${style.colorBtn} ${style.white}`}
          onClick={() => {
            changeColor("white")
          }}
        ></button>
      </div>
    </div>
  )
}
