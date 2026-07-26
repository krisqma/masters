import type { AppStatus } from '../types'

export interface WilgusVisualState {
  src: string
  label: string
  ringClass: string
}

const asset = (name: string) => `/wilgus/${name}`

export const WILGUS_BY_STATUS: Record<AppStatus, WilgusVisualState> = {
  IDLE: {
    src: asset('wilgus_neutral.png'),
    label: 'Gotowy — kliknij, aby mówić',
    ringClass: 'wilgus-ring-idle',
  },
  LISTENING_WAKE_WORD: {
    src: asset('wilgus_neutral.png'),
    label: 'Powiedz „hej wilguś” albo kliknij mnie',
    ringClass: 'wilgus-ring-idle',
  },
  LISTENING_FOLLOW_UP: {
    src: asset('wilgus_macha.gif'),
    label: 'Słucham dalej — możesz mówić bez wake worda',
    ringClass: 'wilgus-ring-follow',
  },
  RECORDING: {
    src: asset('wilgus_zdziwiony.gif'),
    label: 'Nagrywam — kliknij, aby zakończyć',
    ringClass: 'wilgus-ring-recording',
  },
  PROCESSING: {
    src: asset('wilgus_pomysl.gif'),
    label: 'Myślę…',
    ringClass: 'wilgus-ring-processing',
  },
  SPEAKING: {
    src: asset('wilgus_gada.gif'),
    label: 'Mówię…',
    ringClass: 'wilgus-ring-speaking',
  },
}
