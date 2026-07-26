/** Short filler phrases played instantly while the model thinks. */
export interface AckPhrase {
  id: string
  text: string
  /** Path under Vite public/, e.g. /acks/01.mp3 */
  src: string
}

export const ACK_PHRASES: AckPhrase[] = [
  { id: '01', text: 'Sekundka.', src: '/acks/01.mp3' },
  { id: '02', text: 'Już sprawdzam w danych.', src: '/acks/02.mp3' },
  { id: '03', text: 'Hmm, spojrzę.', src: '/acks/03.mp3' },
  { id: '04', text: 'Okej, sprawdzam.', src: '/acks/04.mp3' },
  { id: '05', text: 'Zaraz wrócę z odpowiedzią.', src: '/acks/05.mp3' },
  { id: '06', text: 'Aha, moment.', src: '/acks/06.mp3' },
  { id: '07', text: 'Patrzę na kontekst.', src: '/acks/07.mp3' },
  { id: '08', text: 'Już się tym zajmuję.', src: '/acks/08.mp3' },
  { id: '09', text: 'Daj mi chwilę.', src: '/acks/09.mp3' },
  { id: '10', text: 'Sprawdzam.', src: '/acks/10.mp3' },
  { id: '11', text: 'O, zaraz zobaczę.', src: '/acks/11.mp3' },
  { id: '12', text: 'Jasne, sprawdzam dane.', src: '/acks/12.mp3' },
  { id: '13', text: 'Hmm, zaraz powiem.', src: '/acks/13.mp3' },
  { id: '14', text: 'Moment, zaglądam.', src: '/acks/14.mp3' },
  { id: '15', text: 'Już patrzę.', src: '/acks/15.mp3' },
  { id: '16', text: 'Rozumiem, sprawdzam.', src: '/acks/16.mp3' },
  { id: '17', text: 'Chwileczkę.', src: '/acks/17.mp3' },
  { id: '18', text: 'Zaraz to ogarnę.', src: '/acks/18.mp3' },
  { id: '19', text: 'Okej, moment.', src: '/acks/19.mp3' },
  { id: '20', text: 'Patrzę w dane domu.', src: '/acks/20.mp3' },
  { id: '21', text: 'Już wracam z info.', src: '/acks/21.mp3' },
  { id: '22', text: 'Aha, sprawdzę.', src: '/acks/22.mp3' },
  { id: '23', text: 'Dobra, zaraz.', src: '/acks/23.mp3' },
  { id: '24', text: 'Zaglądam do czujników.', src: '/acks/24.mp3' },
]
