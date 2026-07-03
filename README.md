## 📌 안내 
### 컴퓨터 켜서 작업 시작할 때: 
무조건 git pull origin main을 먼저 해서 다른 팀원이 고친 최신 코드를 내 노트북으로 가져옵니다~!

### 내 기능 구현이 끝났을 때:
git add .
git commit -m "Feat: Implement FastAPI SQLite connection"
git push origin main

## 📌 커밋 메시지 기본 구조
태그: 제목 (영어 대문자로 시작하지 않고, 동사 원형 사용 권장 혹은 명확한 한글 표현)
- 본문 (생략 가능, 왜 변경했는지 설명이 필요할 때 작성)

## 📌 자주 사용하는 태그(Tag) 종류
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정 (README.md, 주석 등)
- `style`: 코드 포맷팅, 세미콜론 누락 등 (코드 자체의 로직 변경이 없는 경우)
- `refactor`: 코드 리팩토링 (기능 추가나 버그 수정이 없는 구조 개선)
- `test`: 테스트 코드 추가 및 수정
- `chore`: 빌드 업무 수정, 패키지 매니저 설정, 프로젝트 설정 변경 (.gitignore 등)

**올바른 예시:**
- `feat: 상품 검색 필터링 기능 추가`
- `fix: 로그인 화면에서 비밀번호 입력 시 앱이 꺼지는 오류 수정`
- `docs: README 파일에 설치 가이드 업데이트`

## 📌 PR 템플릿 양식
### 💡 관련 이슈 / 작업 내용
- 어떤 기능을 구현했는지 간략하게 적어주세요.
- 예: 로그인 페이지 마크업 및 카카오 로그인 API 연동

### 🛠️ 주요 변경 사항
- 핵심적으로 변경된 파일이나 로직을 적어주세요.
- `src/components/Login.tsx`: UI 컴포넌트 구현
- `src/hooks/useAuth.ts`: 인증 상태 관리 커스텀 훅 추가

### 📷 스크린샷 (선택 사항)
*UI 변경이 있는 경우 캡처 사진이나 GIF를 첨부하면 리뷰어가 이해하기 훨씬 쉽습니다.*

### ❓ 리뷰어에게 바라는 점 / 특이 사항
- 충돌이 우려되는 부분이나, 로직 피드백을 받고 싶은 부분을 적어주세요.

## 📌 브랜치 이름 명명 규칙 (Branch Naming)
브랜치 이름만 보고도 어떤 작업이 진행 중인지 파악할 수 있도록 접두사를 사용합니다.

- **형식:** `접두사/기능-요약` (또는 `접두사/이름-기능`)
- **접두사 종류:**
    - `feature/` : 새로운 기능 개발 (예: `feature/signup-form`)
    - `fix/` : 버그 수정 (예: `fix/token-error`)
    - `docs/` : 문서 작업 (예: `docs/readme`)
    - `refactor/` : 전면적인 코드 구조 개선 (예: `refactor/api-layer`)
---
## 📌 Github 작업 프로세스 
### 1. 최신 코드 가져오기 (시작 전 필수)
- git checkout main
- git pull origin main

### 2. 새 브랜치 만들고 이동하기
**브랜치 생성과 이동을 동시에 하기**
- git checkout -b 기능이름-또는-이슈번호

### 3. 코드 수정 및 상태 확인
- git status

### 4. 스테이징 및 커밋 (Commit)
**변경된 모든 파일을 올릴 때**
- git add .

**특정 파일만 올릴 때**
- git add 파일경로/파일명.확장자

**커밋 메시지 작성하기**
- git commit -m "feat: 로그인 기능 구현"

### 5. 원격 저장소에 푸시 (Push)
- git push -u origin 내-브랜치-이름

### 6. Pull Request (PR) 생성하기
푸시가 성공적으로 완료되면, GitHub 웹사이트에 접속
- 해당 레포지토리(Repository) 페이지로 이동하면 상단에 "Compare & pull request"라는 초록색 버튼이 자동으로 떠 있는 것을 볼 수 있음.
- 버튼이 보이지 않는다면 Pull requests 탭으로 이동해 "New pull request"를 누른 뒤, 내가 만든 브랜치를 선택함.
- base 브랜치(코드가 합쳐질 곳, 예: main)와 compare 브랜치(내가 작업한 브랜치)가 올바르게 선택되었는지 확인함.
- 제목과 어떤 내용을 수정했는지 본문을 상세히 적은 후 "Create pull request"를 누르면 완성!
---
## 📌 기타
### 원격 저장소(origin)의 브랜치 삭제
- git push origin --delete 브랜치이름

### 원격에서 지워진 브랜치 목록을 로컬에도 반영하여 정리
- git fetch --prune



