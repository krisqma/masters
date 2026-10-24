// Editable figures: author native shapes/text, validate, then import PPTX into Canva.
// Run with the bundled Node runtime and RUNTIME_NODE_MODULES set.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const workspaceDir=process.cwd();
const staging=path.join(workspaceDir,process.argv[2]||'.tmp-schematy-v3');
const runtimeModules=process.env.RUNTIME_NODE_MODULES;
if(!runtimeModules)throw new Error('Set RUNTIME_NODE_MODULES to the bundled Node modules directory.');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtimeModules,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const skillDir='/Users/krzysztof/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.22136/skills/presentations';
const python='/Users/krzysztof/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const C={ink:'#252A30',muted:'#727980',edge:'#B8BEC4',line:'#717A83',soft:'#F3F4F5',white:'#FFFFFF',accent:'#EDF3F8',accentEdge:'#778C9F'};

function text(s,name,content,x,y,w,h,{size=34,bold=false,color=C.ink}={}){
 const t=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=content;t.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'center',verticalAlignment:'middle',insets:0,wrap:'none',autoFit:'none'};
 return t;
}
function box(s,name,content,x,y,w,h,{fill=C.white,stroke=C.edge,weight=1.7,radius=18,size=34,bold=false,color=C.ink}={}){
 const b=s.shapes.add({name,geometry:'roundRect',borderRadius:radius,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:weight}});
 b.text=content;b.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'center',verticalAlignment:'middle',insets:8,wrap:'none',autoFit:'none'};
 return b;
}
function line(s,x1,y1,x2,y2,{primary=false,arrow=false}={}){
 return s.shapes.add({geometry:'connector',kind:'straight',position:{left:x1,top:y1,width:x2-x1,height:y2-y1},line:{fill:primary?C.line:C.edge,width:primary?2.4:1.7,style:'solid'},cap:'round',join:'round',...(arrow?{tail:{type:'triangle',width:'lg',length:'lg'}}:{})});
}
async function save(p,name,width,height){
 await fs.mkdir(path.join(staging,'final'),{recursive:true});
 const candidatePath=path.join(staging,name+'.candidate.pptx');
 await(await PresentationFile.exportPptx(p)).save(candidatePath);
 await finalizePresentation({workspaceDir,candidatePath,finalPath:path.join(staging,'final',name+'.pptx'),pythonExecutable:python,integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu',`${width*9525},${height*9525}`,'--validate-heading-fit'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Arial']},requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],verifyArtifactToolImport:true,receiptPath:path.join(staging,name+'.validation.json')});
 const png=await p.export({slide:p.slides.items[0],format:'png',scale:1.5});
 await fs.writeFile(path.join(staging,name+'.png'),new Uint8Array(await png.arrayBuffer()));
 console.log(name+' ready');
}

const p1=Presentation.create({slideSize:{width:1440,height:710}});
const s1=p1.slides.add();s1.background.fill=C.white;
text(s1,'ai','Sztuczna inteligencja',470,20,500,70,{size:42,bold:true});
line(s1,720,96,720,126,{primary:true});line(s1,620,126,1240,126);
line(s1,620,126,620,161,{primary:true});line(s1,1240,126,1240,161);
box(s1,'genai','Generatywna sztuczna inteligencja (GenAI)',265,161,710,78,{fill:C.soft,stroke:'none',weight:0,radius:21,size:31});
text(s1,'other-ai','Inne obszary AI',1090,161,300,78,{size:29,color:C.muted});
line(s1,620,239,620,280,{primary:true});line(s1,160,280,1265,280);
for(const x of [160,620,990,1265])line(s1,x,280,x,314,{primary:x===620});
text(s1,'obrazy','Obrazy',35,314,250,64,{size:31,color:C.muted});
text(s1,'tekst','Tekst',500,314,240,64,{size:34,bold:true});
text(s1,'audio','Dźwięk i mowa',835,314,310,64,{size:31,color:C.muted});
text(s1,'other-genai','Inne',1160,314,210,64,{size:31,color:C.muted});
line(s1,620,384,620,436,{primary:true});
box(s1,'lm','Generatywne modele językowe',340,436,560,78,{fill:C.soft,stroke:'none',weight:0,radius:20,size:34});
line(s1,620,514,620,556,{primary:true});line(s1,475,556,775,556);
line(s1,475,556,475,601);line(s1,775,556,775,601,{primary:true});
box(s1,'llm','LLM',365,601,220,78,{radius:21,size:37,stroke:C.edge,weight:1.8});
box(s1,'slm','SLM *',665,601,220,78,{fill:C.accent,stroke:C.accentEdge,weight:2,radius:21,size:39,bold:true});
s1.speakerNotes.text='Opracowanie własne. Treść zgodna z zaakceptowaną uproszczoną wersją. Wybrane gałęzie nie tworzą pełnej, rozłącznej klasyfikacji AI. Gwiazdka przy SLM objaśniona pod podpisem rysunku w pracy: główny obszar badań — SLM w systemie IoT.';
if(!process.argv.includes('--flow-only')) await save(p1,'zakres-pracy-ai',1440,710);

const p2=Presentation.create({slideSize:{width:1600,height:310}});
const s2=p2.slides.add();s2.background.fill=C.white;
const blocks=[['query','Zapytanie',10,210,false],['vad','VAD',338,120,true],['stt','STT',568,120,true],['lm','Model językowy',804,240,true],['tts','TTS',1162,120,true],['answer','Odpowiedź',1400,190,false]];
for(const [id,content,x,w,framed] of blocks){
 if(framed)box(s2,id,content,x,180,w,94,{fill:id==='lm'?C.soft:C.white,stroke:C.line,weight:id==='lm'?1.9:1.6,radius:20,size:id==='lm'?32:36});
 else text(s2,id,content,x,180,w,94,{size:32});
}
for(let i=0;i<5;i++){
 const a=blocks[i],b=blocks[i+1],left=a[2]+a[3]+6,right=b[2]-6;
 line(s2,left,230,right,230,{primary:true,arrow:true});
 text(s2,'format-'+i,i<2||i===4?'audio':'tekst',left,185,right-left,34,{size:28,color:C.muted});
}
text(s2,'context','Kontekst',804,23,240,58,{size:31});
line(s2,924,88,924,173,{primary:true,arrow:true});
s2.speakerNotes.text='Opracowanie własne. Bloki przedstawiają VAD, STT, model językowy i TTS. Zapytanie, odpowiedź oraz kontekst pozostają poza ramkami jako dane wejściowe i wyjściowe. Formaty audio i tekst podano przy strzałkach.';
await save(p2,'tor-interakcji-glosowej',1600,310);
