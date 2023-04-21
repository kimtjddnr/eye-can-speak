import { useCallback, useEffect, useState } from "react"
import style from "../../styles/board/category.module.css"
import { getCategory } from "../../services/commonApi"

export default function Category({ changeCategory }: { changeCategory: any }) {
  const [categoryList, setCategoryList] = useState<string[]>([])

  const setCategory = useCallback(async () => {
    const response = await getCategory()
    setCategoryList(() => [response.data.draw_category])
  }, [])

  useEffect(() => {
    setCategory()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return (
    <div className={style.container}>
      {categoryList.map((item, index) => (
        <button
          className={style.item}
          onClick={() => {
            changeCategory({ index })
          }}
        >
          {categoryList[index]}
        </button>
      ))}
    </div>
  )
}
