import { AxiosResponse } from "axios"
import customAxios from "./api"

export async function getWords() {
  const response: AxiosResponse = await customAxios.get(`/write/history/`)
  return response
}

export async function saveWord(payload: string) {
  const response: AxiosResponse = await customAxios.post(`/write/`, {
    write_content: payload,
  })
  return response
}