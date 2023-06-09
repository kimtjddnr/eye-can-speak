import { useCallback, useEffect, useState } from "react"
import { getCategory } from "../../services/commonApi"
import { getSubject } from "../../services/selectSubject"
import style from "../../styles/selectDrawing/subjectDraw.module.css"
import Subject from "../../assets/image/SubjectDraw.png"
import resetImg from "../../assets/image/reset.png"
import backImg from "../../assets/image/left.png"
import { Link } from "react-router-dom"

export default function SelectSubjectDraw() {
  interface categoryTypes {
    categoryNo: number
    categoryNM: string
  }
  interface subjectTypes {
    subjectNo: number
    subjectNM: string
  }
  // 클릭하면 카테고리 보여주기
  const handleHover = () => {
    const subjectCard = document.querySelector("#subjectCard")
    subjectCard?.classList.add(`${style.hover}`)
  }
  // 서브젝트 리스트
  const [subjects, setSubjects] = useState<subjectTypes[]>([])

  // 랜덤으로 나온 서브젝트
  const [subject, setSubject] = useState<string>("")

  // 카테고리 리스트
  const [category, setCategory] = useState<categoryTypes[]>([])

  // 선택된 카테고리
  const [selectedCategory, setSelectedCategory] = useState<number>(-1)

  // isSelectCategory 가 true면 랜덤 서브젝트를 보여줌
  const [isSelectCategory, SetIsSelectCategory] = useState(false)

  const showSubject = () => {
    SetIsSelectCategory((prev) => !prev)
  }

  // 랜덤으로 subject선택
  const getRandomSubject = () => {
    const randomIndex = Math.floor(Math.random() * subjects.length)
    const randomSubject = subjects[randomIndex]
    if (randomSubject.subjectNM === subject) {
      getRandomSubject()
      return
    }
    setSubject(randomSubject.subjectNM)
  }

  // Category 선택하고 Category안의 subject 부르기

  const selectCategory = async (categoryNum: number) => {
    setSelectedCategory(categoryNum)
    try {
      const response = await getSubject(categoryNum)
      const item = response.data

      showSubject()
      setSubjects(() => [...item])
    } catch (error: any) {
      console.log(error)
    }
  }

  //Category 부르기
  const loadCategory = useCallback(async () => {
    try {
      const response = await getCategory()

      const item = response.data

      setCategory(item)
    } catch (error: any) {
      console.log(error)
    }
  }, [setCategory])

  // category 변경되면 랜덤으로 subject하나 고르는 함수 호출
  useEffect(() => {
    if (subjects.length > 0) {
      getRandomSubject()
    }
  }, [subjects])

  // 렌더링 시 카테고리 가져오기
  useEffect(() => {
    loadCategory()
  }, [])

  return (
    <div className={style.card} id='subjectCard'>
      <h3 className={style.title}>주제선택하기</h3>
      {/* 카드 앞면 */}
      <div
        onClick={handleHover}
        className={style.front}
        style={{ backgroundImage: `url(${Subject})` }}
      ></div>
      {isSelectCategory ? (
        <div className={style.back}>
          {/* subject고르기 */}
          <div className={style.subjectItem}>
            <p>{subject && subject}</p>
            <Link to={`/drawing/${selectedCategory}`} className={style.draw}>
              그리기
            </Link>
            <div className={style.buttonBox}>
              {/* 뒤로가기버튼 카테고리 선택 */}
              <img onClick={showSubject} src={backImg} alt='back' />
              {/* 새로고침 버튼 새 subject 가져오기 */}
              <img onClick={getRandomSubject} src={resetImg} alt='reset' />
            </div>
          </div>
        </div>
      ) : (
        <div
          className={style.back}
          style={{ backgroundImage: `url(${Subject})` }}
        >
          {/* category 고르기 */}
          {category &&
            category.map((item, idx) =>
              idx <= 3 ? (
                <div
                  onClick={() => {
                    selectCategory(item.categoryNo)
                  }}
                  className={style.subject}
                  key={idx}
                >
                  {item.categoryNM}
                </div>
              ) : null
            )}
        </div>
      )}
    </div>
  )
}
